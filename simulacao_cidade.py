import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import FuncFormatter

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

# ==============================================================================
# 1. CONFIGURAÇÕES E DADOS INICIAIS
# ==============================================================================

# Parâmetros de Tempo e População Base
ano_inicial = 2014
ano_projecao = 2021
ano_inicio_projecao = 2026   # Início da exibição nos gráficos
anos_historicos = ano_projecao - ano_inicial
anos_projecao = 50
max_idade = 100
pop_inicial_total = 56629
sexo_ratio_f = 0.51

# Dados do Censo 2010 (Estrutura Etária Real)
censo_data_f = {
    '0 a 4 anos': 2459, '5 a 9 anos': 2640, '10 a 14 anos': 2661,
    '15 a 19 anos': 2632, '20 a 24 anos': 2765, '25 a 29 anos': 2512,
    '30 a 39 anos': 4432, '40 a 49 anos': 3421, '50 a 59 anos': 2238,
    '60 a 69 anos': 1587, '70 ou mais': 1678
}
censo_data_m = {
    '0 a 4 anos': 2622, '5 a 9 anos': 2675, '10 a 14 anos': 2820,
    '15 a 19 anos': 2550, '20 a 24 anos': 2774, '25 a 29 anos': 2563,
    '30 a 39 anos': 4348, '40 a 49 anos': 3070, '50 a 59 anos': 1907,
    '60 a 69 anos': 1252, '70 ou mais': 1023
}

# Dados Históricos Reais (IBGE/SIOPS/SIOPE)
dados_historicos = {
    'ano': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'populacao_real': [56629, 57000, 58000, 59000, 60000, 61000, 62000, 62500],
    'pib_real': [
        776670000.52, 744482000.30, 784308000.63, 792337000.75,
        858899000.62, 975936000.10, 969253000.11, 1002105000.38
    ],
    'gasto_saude_real': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 33194847.29, 42141035.96],
    'gasto_educ_real': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 38997905.83, 50890153.21]
}
df_hist = pd.DataFrame(dados_historicos).set_index('ano')

# Parâmetros Econômicos Base
work_start, work_end = 18, 64
desemprego_base = 0.07
produtividade_base = {'agro': 30000, 'industria': 35000, 'servicos': 50000}
participacao_setores = {'agro': 0.0145, 'industria': 0.1398, 'servicos': 0.8457}
percent_escolarizacao = {'fund': 0.98, 'medio': 0.85, 'superior': 0.30}
anos_nivel = {'fund': 9, 'medio': 3, 'superior': 4}
anos_fund = 9

# ==============================================================================
# 2. FUNÇÕES AUXILIARES (Dist, Formatação)
# ==============================================================================

def distribute_age_groups(censo_groups, total_pop_scaled, max_age=100):
    pop_array = np.zeros(max_age + 1)
    total_censo_pop = sum(censo_groups.values())
    if total_censo_pop == 0: return pop_array

    for group_key, pop_censo in censo_groups.items():
        proportion = pop_censo / total_censo_pop
        scaled_group_pop = proportion * total_pop_scaled
        if 'ou mais' in group_key:
            start_age = int(group_key.split(' ')[0])
            end_age = max_age
        else:
            parts = group_key.split(' a ')
            start_age = int(parts[0])
            end_age = int(parts[1].split(' ')[0])
        
        num_years = end_age - start_age + 1
        pop_per_year = scaled_group_pop / num_years
        pop_array[start_age : end_age + 1] = pop_per_year
    
    if pop_array.sum() > 0:
        pop_array = (pop_array / pop_array.sum()) * total_pop_scaled
    return pop_array

def formatar_valor(valor, pos):
    if valor >= 1e9: return f"R$ {valor/1e9:.1f} bi"
    elif valor >= 1e6: return f"R$ {valor/1e6:.1f} mi"
    else: return f"R$ {valor:.0f}"

def formatar_habitantes(valor, pos):
    if valor >= 1e6: return f"{valor/1e6:.2f} mi"
    elif valor >= 1e3: return f"{valor/1e3:.0f} mil"
    else: return f"{valor:.0f}"

def formatar_smart(valor, prefixo="R$ "):
    if valor >= 1e9: return f"{prefixo}{valor/1e9:.2f} bi"
    else: return f"{prefixo}{valor:,.2f}"

# Inicialização das Curvas Demográficas
pop_inicial_f = distribute_age_groups(censo_data_f, pop_inicial_total * sexo_ratio_f, max_idade)
pop_inicial_m = distribute_age_groups(censo_data_m, pop_inicial_total * (1 - sexo_ratio_f), max_idade)

fert_by_age = np.zeros(max_idade + 1)
fert_by_age[15:29] = np.linspace(0.02, 0.12, 14)
fert_by_age[29:50] = np.linspace(0.12 * 0.95, 0.005, 21)

all_ages = np.arange(max_idade + 1)
mort_by_age_base = np.interp(all_ages, [0, 1, 15, 30, 60, 80, 100], [0.012, 0.001, 0.0006, 0.0015, 0.01, 0.07, 0.5])

mig_by_age_dist = np.zeros(max_idade + 1)
mig_by_age_dist[20:41] = 1
mig_by_age_dist /= mig_by_age_dist.sum()
mig_total_base = 150

# ==============================================================================
# 3. MOTOR DE SIMULAÇÃO (COMPLETO E CORRIGIDO)
# ==============================================================================

def run_simulation(fert_scale=1.0, mort_scale=1.0, mig_scale=1.0,
                   fert_increase_pct=0.0, anos_simular=50, pop_f_ini=None, pop_m_ini=None,
                   return_history=False,
                   param_custo_aluno=2500.0, param_gasto_saude_pc=1500.0,
                   invest_educ_pct=0.0, invest_saude_pct=0.0):
    
    fert_local = fert_by_age * fert_scale * (1 + fert_increase_pct)
    mort_local = mort_by_age_base * mort_scale
    
    # Feedback: Investimento em saúde reduz mortalidade
    if invest_saude_pct != 0:
        fator_saude = 1.0 - (invest_saude_pct * 0.2)
        mort_local = mort_local * fator_saude

    mig_total_local = mig_total_base * mig_scale
    pop_f = np.zeros((anos_simular + 1, max_idade + 1))
    pop_m = np.zeros((anos_simular + 1, max_idade + 1))
    pop_f[0, :] = pop_f_ini.copy()
    pop_m[0, :] = pop_m_ini.copy()

    total_pop = [pop_f_ini.sum() + pop_m_ini.sum()]
    pib, gasto_educacao, gasto_saude, nivel_escolaridade_medio_hist = [], [], [], []
    pib_per_capita = []

    # Novas Variáveis para Gráficos
    dependent_ratio_hist = []
    alunos_creche_hist = []  # 0 a 3 anos
    alunos_fundamental_hist = [] # 6 a 14 anos

    # Calibração Inicial do PIB
    pop_work_ini = np.sum(pop_f_ini[work_start:work_end+1] + pop_m_ini[work_start:work_end+1])
    prod_set_ini = sum(np.array(list(produtividade_base.values())) * np.array(list(participacao_setores.values())))
    fator_escala_pib = df_hist['pib_real'].iloc[0] / (pop_work_ini * prod_set_ini)

    # Variáveis de Feedback
    multiplicador_produtividade = 1.0
    bonus_escolarizacao = 0.0
    taxa_desemprego_ant = desemprego_base

    for t in range(anos_simular):
        # Lógica de Feedback
        ganho_produtividade = invest_educ_pct * 0.015
        multiplicador_produtividade *= (1 + ganho_produtividade)
        
        bonus_escolarizacao += (invest_educ_pct * 0.005)
        bonus_escolarizacao = min(0.15, max(-0.10, bonus_escolarizacao))
        
        perc_educ_atual = {k: min(1.0, v + bonus_escolarizacao) for k, v in percent_escolarizacao.items()}

        # Dinâmica Populacional
        popf, popm = pop_f[t].copy(), pop_m[t].copy()
        sob_f, sob_m = popf * (1 - mort_local), popm * (1 - mort_local)
        new_f, new_m = np.zeros_like(popf), np.zeros_like(popm)
        new_f[1:], new_m[1:] = sob_f[:-1], sob_m[:-1]
        
        nasc = np.sum(sob_f * fert_local)
        new_f[0], new_m[0] = nasc * sexo_ratio_f, nasc * (1 - sexo_ratio_f)
        
        mig_in = mig_total_local * mig_by_age_dist
        new_f += mig_in * sexo_ratio_f
        new_m += mig_in * (1 - sexo_ratio_f)
        
        pop_f[t+1], pop_m[t+1] = new_f, new_m
        total_pop.append(new_f.sum() + new_m.sum())

        # Economia
        pop_work = np.sum(new_f[work_start:work_end+1] + new_m[work_start:work_end+1])
        taxa_desemprego_ant = max(0.01, min(0.45, 0.7 * taxa_desemprego_ant + 0.3 * desemprego_base))
        
        prod_setorial = sum(np.array(list(produtividade_base.values())) * np.array(list(participacao_setores.values())))
        pib_atual = (pop_work * (1 - taxa_desemprego_ant)) * prod_setorial * fator_escala_pib * multiplicador_produtividade
        
        pib.append(pib_atual)
        pib_per_capita.append(pib_atual / (new_f.sum() + new_m.sum()) if (new_f.sum() + new_m.sum()) > 0 else 0)

        # Cálculos Demográficos Extras
        pop_jovem = new_f[0:15].sum() + new_m[0:15].sum() # 0-14
        pop_ativa = new_f[15:65].sum() + new_m[15:65].sum() # 15-64
        pop_idosa = new_f[65:max_idade+1].sum() + new_m[65:max_idade+1].sum() # 65+
        
        dep_ratio = (pop_jovem + pop_idosa) / pop_ativa * 100 if pop_ativa > 0 else 0
        dependent_ratio_hist.append(dep_ratio)

        alunos_creche = new_f[0:4].sum() + new_m[0:4].sum()
        alunos_fundamental = new_f[6:15].sum() + new_m[6:15].sum()
        alunos_creche_hist.append(alunos_creche)
        alunos_fundamental_hist.append(alunos_fundamental)

        # Gastos Sociais
        idade_inicio_fund = 6
        alunos = new_f[idade_inicio_fund:idade_inicio_fund+anos_fund].sum() * perc_educ_atual['fund']
        gasto_e = alunos * param_custo_aluno * (1 + invest_educ_pct)
        gasto_s = (new_f.sum() + new_m.sum()) * param_gasto_saude_pc * (1 + invest_saude_pct)
        
        gasto_educacao.append(gasto_e)
        gasto_saude.append(gasto_s)

        # Escolaridade Média
        escolaridade_base = 7.0 
        nivel_educ = escolaridade_base * multiplicador_produtividade
        nivel_educ = min(16.0, nivel_educ) 
        nivel_escolaridade_medio_hist.append(nivel_educ)

    outputs = {
        "total_pop": np.array(total_pop), "pib": np.array(pib), 
        "pib_per_capita": np.array(pib_per_capita),
        "gasto_saude": np.array(gasto_saude), "gasto_educ": np.array(gasto_educacao),
        "nivel_escolaridade_medio": np.array(nivel_escolaridade_medio_hist),
        "dep_ratio": np.array(dependent_ratio_hist),
        "alunos_creche": np.array(alunos_creche_hist),
        "alunos_fundamental": np.array(alunos_fundamental_hist)
    }
    if return_history:
        outputs.update({"pop_f": pop_f, "pop_m": pop_m})
    return outputs

# ==============================================================================
# 4. CALIBRAÇÃO AUTOMÁTICA
# ==============================================================================

def calibration_objective(x):
    fert_s, mort_s, mig_s, custo_a, saude_pc = x
    out = run_simulation(fert_s, mort_s, mig_s, anos_simular=anos_historicos, 
                         pop_f_ini=pop_inicial_f, pop_m_ini=pop_inicial_m,
                         param_custo_aluno=custo_a, param_gasto_saude_pc=saude_pc)
    
    sim_pop = out['total_pop'][1:]
    sim_pib = out['pib'][:anos_historicos]
    
    real_pop = df_hist['populacao_real'].values[1:]
    real_pib = df_hist['pib_real'].values[1:]
    
    idx_dados = df_hist['gasto_saude_real'].dropna().index - ano_inicial - 1
    sim_saude = np.array(out['gasto_saude'])[idx_dados]
    sim_educ = np.array(out['gasto_educ'])[idx_dados]
    real_saude = df_hist['gasto_saude_real'].dropna().values
    real_educ = df_hist['gasto_educ_real'].dropna().values

    mae_pop = np.mean(np.abs(sim_pop - real_pop))
    mae_pib = np.mean(np.abs(sim_pib - real_pib))
    mae_saude = np.mean(np.abs(sim_saude - real_saude))
    mae_educ = np.mean(np.abs(sim_educ - real_educ))

    return (0.2 * (mae_pop/real_pop.mean()) + 0.2 * (mae_pib/real_pib.mean()) +
            0.3 * (mae_saude/real_saude.mean()) + 0.3 * (mae_educ/real_educ.mean()))

# ==============================================================================
# 5. INTERFACE STREAMLIT
# ==============================================================================

st.set_page_config(layout="wide")
st.title("Simulador: Fertilidade e Impacto Econômico")
st.write(f"Período histórico: {ano_inicial} a {ano_projecao}. Projeção até {ano_projecao + anos_projecao}.")

st.sidebar.header("Ajustes de Cenário")
fert_increase_pct = st.sidebar.slider("Aumento na taxa de natalidade (%)", -50, 100, 0) / 100
invest_educ_pct = st.sidebar.slider("Variação no Investimento em Educação (%)", -50, 100, 0) / 100
invest_saude_pct = st.sidebar.slider("Variação no Investimento em Saúde (%)", -50, 100, 0) / 100
ano_usuario = st.sidebar.slider("Selecione o ano da projeção", min_value=ano_inicio_projecao,
                                 max_value=ano_projecao+anos_projecao, value=2031)

# Execução da Calibração (Background)
initial_guess = [1.0, 1.0, 1.0, 4500.0, 600.0]
param_bounds = [(0.5, 1.5), (0.7, 1.3), (0.2, 5.0), (3500.0, 12000.0), (300.0, 2000.0)]

fert_opt, mort_opt, mig_opt, custo_opt, saude_opt = initial_guess
if minimize is not None:
    res = minimize(calibration_objective, initial_guess, method='L-BFGS-B', bounds=param_bounds)
    if res.success:
        fert_opt, mort_opt, mig_opt, custo_opt, saude_opt = res.x
        st.sidebar.success("Calibração concluída!")
    else:
        st.sidebar.warning("Usando parâmetros padrão.")

# Execução das Simulações (Completa)
out_hist = run_simulation(fert_opt, mort_opt, mig_opt, anos_simular=anos_historicos, 
                          pop_f_ini=pop_inicial_f, pop_m_ini=pop_inicial_m, return_history=True,
                          param_custo_aluno=custo_opt, param_gasto_saude_pc=saude_opt)

anos_proj_user = ano_usuario - ano_projecao
out_proj = run_simulation(fert_opt, mort_opt, mig_opt, fert_increase_pct=fert_increase_pct,
                          anos_simular=anos_proj_user, pop_f_ini=out_hist['pop_f'][-1], 
                          pop_m_ini=out_hist['pop_m'][-1], param_custo_aluno=custo_opt, 
                          param_gasto_saude_pc=saude_opt, invest_educ_pct=invest_educ_pct, 
                          invest_saude_pct=invest_saude_pct, return_history=True)

# --- Fatiamento dos dados para começar em 2026 ---
ano_sim_start = ano_projecao + 1 # 2022
indice_corte = ano_inicio_projecao - ano_sim_start # 2026 - 2022 = 4

# Eixo X: anos de 2026 até ano_usuario
anos_sim_cortado = np.arange(ano_inicio_projecao, ano_usuario + 1)

# Fatiamento dos dados Y:
pop_proj_slice = out_proj['total_pop'][indice_corte + 1:]
pib_slice = out_proj['pib'][indice_corte:]
pib_pc_slice = out_proj['pib_per_capita'][indice_corte:]
gasto_educ_slice = out_proj['gasto_educ'][indice_corte:]
gasto_saude_slice = out_proj['gasto_saude'][indice_corte:]
escolaridade_slice = out_proj['nivel_escolaridade_medio'][indice_corte:]
dep_ratio_slice = out_proj['dep_ratio'][indice_corte:]
alunos_creche_slice = out_proj['alunos_creche'][indice_corte:]
alunos_fundamental_slice = out_proj['alunos_fundamental'][indice_corte:]

# --- Métricas de Precisão (Erro Geral) ---
sim_pop_h = out_hist['total_pop'][1:anos_historicos+1]
real_pop_h = df_hist['populacao_real'].values[1:]
sim_pib_h = out_hist['pib'][:anos_historicos]
real_pib_h = df_hist['pib_real'].values[1:]

err_pop = np.mean(np.abs(sim_pop_h - real_pop_h) / real_pop_h * 100)
err_pib = np.mean(np.abs(sim_pib_h - real_pib_h) / real_pib_h * 100)

idx_v = df_hist['gasto_saude_real'].dropna().index - ano_inicial - 1
sim_sau_h = np.array(out_hist['gasto_saude'])[idx_v]
real_sau_h = df_hist['gasto_saude_real'].dropna().values
sim_edu_h = np.array(out_hist['gasto_educ'])[idx_v]
real_edu_h = df_hist['gasto_educ_real'].dropna().values

err_sau = np.mean(np.abs(sim_sau_h - real_sau_h) / real_sau_h * 100)
err_edu = np.mean(np.abs(sim_edu_h - real_edu_h) / real_edu_h * 100)
err_total = (err_pop + err_pib + err_sau + err_edu) / 4

st.markdown("---")
st.subheader("Precisão do Modelo (Calibragem 2015-2021)")
col_t, _ = st.columns([1, 3])
col_t.metric("Erro Global Médio", f"{err_total:.2f}%")

st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("População Total", f"{out_proj['total_pop'][-1]:,.0f} habitantes", help="Total projetado")
c2.metric("PIB Total", formatar_smart(out_proj['pib'][-1]), help="PIB Real projetado")
c3.metric("PIB per Capita", formatar_smart(out_proj['pib_per_capita'][-1]))

st.markdown("### Indicadores Sociais")
c4, c5, c6 = st.columns(3)
c4.metric("Gasto Educação", formatar_smart(out_proj['gasto_educ'][-1]))
c5.metric("Gasto Saúde", formatar_smart(out_proj['gasto_saude'][-1]))
c6.metric("Escolaridade Média", f"{out_proj['nivel_escolaridade_medio'][-1]:.2f} anos", help="Média estimada de anos de estudo")
st.markdown("---")

# ==============================================================================
# 6. GRÁFICOS
# ==============================================================================

st.header(f"Evolução da População")
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(anos_sim_cortado, pop_proj_slice, label=f"População (Cenário Atual)")
ax.set_xlabel("Ano")
ax.set_ylabel("População")
ax.grid(True, alpha=0.3)
ax.legend()
ax.yaxis.set_major_formatter(FuncFormatter(formatar_habitantes))
st.pyplot(fig)

st.header(f"PIB Total e Per Capita")
fig2, ax1 = plt.subplots(figsize=(10,6))
c_pib, c_pc = 'tab:blue', 'tab:red'
ax1.plot(anos_sim_cortado, pib_slice, color=c_pib, label='PIB Total')
ax1.set_xlabel("Ano")
ax1.set_ylabel("PIB Total", color=c_pib)
ax1.tick_params(axis='y', labelcolor=c_pib)
ax1.yaxis.set_major_formatter(FuncFormatter(formatar_valor))
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(anos_sim_cortado, pib_pc_slice, color=c_pc, linestyle='--', label='PIB per Capita')
ax2.set_ylabel("PIB per Capita", color=c_pc)
ax2.tick_params(axis='y', labelcolor=c_pc)
ax2.yaxis.set_major_formatter(FuncFormatter(formatar_valor))
fig2.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
st.pyplot(fig2)

# --- GRÁFICO 3: Comparativo de Custo Total (Saúde vs. Educação) ---
st.header(f"Comparativo de Custo Total (Saúde vs. Educação)")
fig3, ax3 = plt.subplots(figsize=(10,6))
ax3.plot(anos_sim_cortado, gasto_educ_slice, label="Educação", color='blue')
ax3.plot(anos_sim_cortado, gasto_saude_slice, label="Saúde", color='red', linestyle="--")
ax3.set_xlabel("Ano")
ax3.set_ylabel("Valor (R$)")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(FuncFormatter(formatar_valor))
st.pyplot(fig3)

# --- GRÁFICO 4: Evolução da Escolaridade Média ---
st.header(f"Evolução da Escolaridade Média")
fig4, ax4 = plt.subplots(figsize=(10,6))
ax4.plot(anos_sim_cortado, escolaridade_slice, label="Anos de Estudo", color='green')
ax4.set_xlabel("Ano")
ax4.set_ylabel("Anos")
ax4.grid(True, alpha=0.3)
ax4.legend()
st.pyplot(fig4)

# --- GRÁFICO 5: Coeficiente de Dependência Total ---
st.header(f"Coeficiente de Dependência Total")
st.caption("Proporção de dependentes (0-14 e 65+ anos) por 100 pessoas em idade ativa (15-64 anos).")
fig_dep, ax_dep = plt.subplots(figsize=(10,6))
ax_dep.plot(anos_sim_cortado, dep_ratio_slice, label="Dependência Total", color='purple')
ax_dep.set_xlabel("Ano")
ax_dep.set_ylabel("Dependentes por 100 Ativos")
ax_dep.grid(True, alpha=0.3)
ax_dep.legend()
st.pyplot(fig_dep)

# --- GRÁFICO 6: Demanda por Vagas em Educação ---
st.header(f"Demanda por Vagas em Educação (0 a 14 anos)")
st.caption("Evolução da população na faixa etária escolar (0-3 anos e 6-14 anos), indicando demanda por infraestrutura.")
fig_edu_dem, ax_edu_dem = plt.subplots(figsize=(10,6))
ax_edu_dem.plot(anos_sim_cortado, alunos_creche_slice, label="0-3 anos (Creche)", color='orange')
ax_edu_dem.plot(anos_sim_cortado, alunos_fundamental_slice, label="6-14 anos (Ensino Fundamental)", color='brown')
ax_edu_dem.set_xlabel("Ano")
ax_edu_dem.set_ylabel("População")
ax_edu_dem.grid(True, alpha=0.3)
ax_edu_dem.legend()
ax_edu_dem.yaxis.set_major_formatter(FuncFormatter(formatar_habitantes))
st.pyplot(fig_edu_dem)

# --- GRÁFICO 7: Pirâmide Etária (CORRIGIDO O ERRO DE SHAPE) ---
st.header(f"Pirâmide Etária: Projeção para o Ano {ano_usuario}")

# 1. Pegar a última população projetada
pop_f_final = out_proj['pop_f'][-1]
pop_m_final = out_proj['pop_m'][-1]

# 2. Definir rótulos das faixas etárias (21 faixas: 0-4 até 100+)
labels = [f'{i}-{i+4}' for i in range(0, 100, 5)] + ['100+']

# 3. Agrupar dados populacionais para corresponder exatamente aos rótulos (21 grupos)
pop_f_groups = [pop_f_final[i:i+5].sum() for i in range(0, 100, 5)]
pop_f_groups.append(pop_f_final[100:].sum()) # Adiciona o grupo 100+

pop_m_groups = [pop_m_final[i:i+5].sum() for i in range(0, 100, 5)]
pop_m_groups.append(pop_m_final[100:].sum()) # Adiciona o grupo 100+

# 4. Plotar (Agora com garantia de shapes iguais)
fig_piramide, ax_piramide = plt.subplots(figsize=(10, 6))

# Inverter a população masculina (para a esquerda)
ax_piramide.barh(labels, -np.array(pop_m_groups), color='skyblue', label='Masculino')
ax_piramide.barh(labels, pop_f_groups, color='pink', label='Feminino')

# Formatação
ax_piramide.set_xlabel("População")
ax_piramide.set_ylabel("Idade")
ax_piramide.set_title(f"Estrutura Etária de População em {ano_usuario}")
ax_piramide.ticklabel_format(style='plain', axis='x')
# Formata o eixo X para mostrar positivo dos dois lados
ax_piramide.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: formatar_habitantes(abs(x), pos)))
ax_piramide.grid(True, alpha=0.3, axis='x')

# Ajustar os limites do eixo X para centralizar
max_pop_abs = max(max(pop_f_groups), max(pop_m_groups))
ax_piramide.set_xlim(-max_pop_abs * 1.2, max_pop_abs * 1.2)
ax_piramide.legend()
st.pyplot(fig_piramide)

st.info(f"Simulação feita a partir de 2022. Os dados históricos anteriores a 2022 foram usados para calibrar o modelo. Os gráficos mostram a projeção a partir do ano de {ano_inicio_projecao}.")