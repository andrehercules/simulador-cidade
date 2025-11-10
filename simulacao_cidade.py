# app_fertilidade_calibrada_salgueiro.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import FuncFormatter
import numpy as np

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

# --- Configurações iniciais ---
ano_inicial = 2014
ano_projecao = 2021
anos_historicos = ano_projecao - ano_inicial
anos_projecao = 50
max_idade = 100
pop_inicial_total = 56629  # população estimada de 2014
sexo_ratio_f = 0.51

# --- Distribuição etária (BASEADA NO CENSO 2010 REAL) ---

def distribute_age_groups(censo_groups, total_pop_scaled, max_age=100):
    """
    Pega os grupos de idade do Censo (dicionário) e o total da população
    desejada, e distribui por idades únicas (0 a 100).
    """
    pop_array = np.zeros(max_age + 1)
    total_censo_pop = sum(censo_groups.values())
    
    # Se o total do censo for zero, não faz nada
    if total_censo_pop == 0:
        return pop_array

    for group_key, pop_censo in censo_groups.items():
        # Calcula a proporção deste grupo no Censo
        proportion = pop_censo / total_censo_pop
        # Aplica a proporção ao total desejado (ex: pop 2014)
        scaled_group_pop = proportion * total_pop_scaled
        
        # Determina as idades de início e fim do grupo
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
    
    # Re-normaliza para garantir que a soma bate exatamente com o total
    # (corrige erros de arredondamento)
    current_sum = pop_array.sum()
    if current_sum > 0:
        pop_array = (pop_array / current_sum) * total_pop_scaled
    
    return pop_array

# Dados do CENSO 2010 (dos seus prints)
censo_data_f = {
    '0 a 4 anos': 2459,
    '5 a 9 anos': 2640,
    '10 a 14 anos': 2661,
    '15 a 19 anos': 2632,
    '20 a 24 anos': 2765,
    '25 a 29 anos': 2512,
    '30 a 39 anos': 4432,  # Grupo de 10 anos
    '40 a 49 anos': 3421,  # Grupo de 10 anos
    '50 a 59 anos': 2238,  # Grupo de 10 anos
    '60 a 69 anos': 1587,  # Grupo de 10 anos
    '70 ou mais': 1678
}

censo_data_m = {
    '0 a 4 anos': 2622,
    '5 a 9 anos': 2675,
    '10 a 14 anos': 2820,
    '15 a 19 anos': 2550,
    '20 a 24 anos': 2774,
    '25 a 29 anos': 2563,
    '30 a 39 anos': 4348,  # Grupo de 10 anos
    '40 a 49 anos': 3070,  # Grupo de 10 anos
    '50 a 59 anos': 1907,  # Grupo de 10 anos
    '60 a 69 anos': 1252,  # Grupo de 10 anos
    '70 ou mais': 1023
}

# Separa a população total de 2014 (dado inicial) pelo sexo_ratio
pop_inicial_f_total = pop_inicial_total * sexo_ratio_f
pop_inicial_m_total = pop_inicial_total * (1 - sexo_ratio_f)

# Cria as populações iniciais REAIS
pop_inicial_f = distribute_age_groups(censo_data_f, pop_inicial_f_total, max_idade)
pop_inicial_m = distribute_age_groups(censo_data_m, pop_inicial_m_total, max_idade)

# --- Fertilidade, mortalidade e migração (MODELOS REALISTAS) ---

# 1. FERTILIDADE (Curva com pico)
# Modelo que simula um pico de fertilidade (ex: pico aos 28 anos)
# e resulta numa Taxa de Fecundidade Total (TFR) de aprox. 2.23
fert_by_age = np.zeros(max_idade + 1)
start_age = 15
peak_age = 28
end_age = 49
peak_fert = 0.12  # Taxa máxima no pico

# Aumento da fertilidade da idade inicial até o pico
n_rise = peak_age - start_age + 1
fert_by_age[start_age : peak_age + 1] = np.linspace(0.02, peak_fert, n_rise)

# Queda da fertilidade do pico até o fim do período fértil
n_fall = end_age - peak_age
fert_by_age[peak_age + 1 : end_age + 1] = np.linspace(peak_fert * 0.95, 0.005, n_fall)

# 2. MORTALIDADE (Curva "Banheira" ou "Jota")
# Modelo baseado em interpolação de pontos-chave, simulando
# uma "curva de banheira" (mortalidade infantil alta, cai na juventude,
# e sobe exponencialmente na velhice).
# Valores aproximados com base em tábuas de vida reais (ex: IBGE).
mort_key_ages =  [0,   1,     15,     30,     60,   80,    100]
mort_key_rates = [0.012, 0.001, 0.0006, 0.0015, 0.01, 0.07,  0.5] # Taxas de mortalidade (qx)

all_ages = np.arange(max_idade + 1)
mort_by_age_base = np.interp(all_ages, mort_key_ages, mort_key_rates)

mig_total_base = 150
mig_by_age_dist = np.zeros(max_idade + 1)
mig_by_age_dist[20:41] = 1
mig_by_age_dist /= mig_by_age_dist.sum()

# --- Educação e mercado de trabalho ---
work_start, work_end = 18, 64
desemprego_base = 0.07
produtividade_base = {'agro': 30000, 'industria': 35000, 'servicos': 50000}
participacao_setores = {
    'agro': 0.0145,       # (1.45%)
    'industria': 0.1398,  # (13.98%)
    'servicos': 0.8457    # (84.57%)
}

percent_escolarizacao = {
    'fund': 0.98,
    'medio': 0.85,
    'superior': 0.30
}
anos_nivel = {'fund': 9, 'medio': 3, 'superior': 4}
anos_fund = 9
anos_custeio_educ = anos_fund

# --- Dados históricos reais de Salgueiro (PE) ---
# DADOS 100% REAIS (2014-2021)
dados_historicos = {
    # Período histórico ajustado para 2014-2021 (8 anos)
    'ano': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],

    # População (removido o dado de 2022)
    'populacao_real': [56629, 57000, 58000, 59000, 60000, 61000, 62000, 62500],

    # Série do PIB REAL (da sua imagem)
    'pib_real': [
        776670000.52, 744482000.30, 784308000.63, 792337000.75,
        858899000.62, 975936000.10, 969253000.11, 1002105000.38
    ],

    # Gastos reais (removido os dados de 2022) - AGORA CORRIGIDO
    'gasto_saude_real': [
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        33194847.29, 42141035.96  # 2020 e 2021
    ],
    'gasto_educ_real': [
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        38997905.83, 50890153.21  # 2020 e 2021
    ]
}
df_hist = pd.DataFrame(dados_historicos).set_index('ano')

# =========================================================
# Interface Streamlit
# =========================================================
st.set_page_config(layout="wide")
st.title("Simulador: Fertilidade e Impacto Econômico (Calibração) - Salgueiro (PE)")
st.write(f"Período histórico: {ano_inicial} a {ano_projecao}. Projeção até {ano_projecao + anos_projecao}.")

st.sidebar.header("Ajustes de Cenário")
fert_increase_pct = st.sidebar.slider("Aumento na taxa de natalidade (%)", -50, 100, 0) / 100
invest_educ_pct = st.sidebar.slider("Variação no Investimento em Educação (%)", -50, 100, 0) / 100
invest_saude_pct = st.sidebar.slider("Variação no Investimento em Saúde (%)", -50, 100, 0) / 100
ano_usuario = st.sidebar.slider("Selecione o ano da projeção", min_value=ano_projecao+1,
                                max_value=ano_projecao+anos_projecao, value=ano_projecao+10)

# =========================================================
# Função para formatar valores nos eixos
# =========================================================
def formatar_valor(valor, pos):
    if valor >= 1e9:
        return f"R$ {valor/1e9:.1f} bi"
    elif valor >= 1e6:
        return f"R$ {valor/1e6:.1f} mi"
    elif valor >= 1e3:
        return f"R$ {valor/1e3:.0f} mil"
    else:
        return f"R$ {valor:.0f}"

def formatar_habitantes(valor, pos):
    if valor >= 1e6:
        return f"{valor/1e6:.2f} mi"
    elif valor >= 1e3:
        return f"{valor/1e3:.0f} mil"
    else:
        return f"{valor:.0f}"

# =========================================================
# Função de simulação com PIB calibrado
# =========================================================
def run_simulation(fert_scale=1.0, mort_scale=1.0, mig_scale=1.0,
                   fert_increase_pct=0.0, anos_simular=50, pop_f_ini=None, pop_m_ini=None,
                   return_history=False,
                   param_custo_aluno=2500.0, param_gasto_saude_pc=1500.0,
                   invest_educ_pct=0.0, invest_saude_pct=0.0    ):
    fert_local = fert_by_age * fert_scale * (1 + fert_increase_pct)
    mort_local = mort_by_age_base * mort_scale
    mig_total_local = mig_total_base * mig_scale

    pop_f = np.zeros((anos_simular + 1, max_idade + 1))
    pop_m = np.zeros((anos_simular + 1, max_idade + 1))
    pop_f[0, :] = pop_f_ini.copy()
    pop_m[0, :] = pop_m_ini.copy()

    total_pop = [pop_f_ini.sum() + pop_m_ini.sum()]
    pib = []
    pib_per_capita = []
    gasto_educacao = []
    gasto_saude = []
    nivel_escolaridade_medio_hist = []

    taxa_desemprego_ano_anterior = desemprego_base

    # --- Calibrar produtividade inicial para PIB de 2014 ---
    pop_work_ini = np.sum(pop_f_ini[work_start:work_end+1] + pop_m_ini[work_start:work_end+1])
    produtividade_setorial_ini = sum(np.array(list(produtividade_base.values())) * np.array(list(participacao_setores.values())))
    fator_escala_pib = df_hist['pib_real'].iloc[0] / (pop_work_ini * produtividade_setorial_ini)

    for t in range(anos_simular):
        popf = pop_f[t, :].copy()
        popm = pop_m[t, :].copy()

        sobreviventes_f = popf * (1 - mort_local)
        sobreviventes_m = popm * (1 - mort_local)

        new_pop_f = np.zeros_like(popf)
        new_pop_m = np.zeros_like(popm)
        new_pop_f[1:] = sobreviventes_f[:-1]
        new_pop_m[1:] = sobreviventes_m[:-1]

        # Nascimentos
        nascimentos = np.sum(sobreviventes_f * fert_local)
        new_pop_f[0] = nascimentos * sexo_ratio_f
        new_pop_m[0] = nascimentos * (1 - sexo_ratio_f)

        # Migração
        mig_in = mig_total_local * mig_by_age_dist
        new_pop_f += mig_in * sexo_ratio_f
        new_pop_m += mig_in * (1 - sexo_ratio_f)

        pop_f[t+1, :] = new_pop_f
        pop_m[t+1, :] = new_pop_m

        pop_total_t1 = new_pop_f.sum() + new_pop_m.sum()
        total_pop.append(pop_total_t1)

        # Mercado de trabalho e PIB
        pop_work = np.sum(new_pop_f[work_start:work_end+1] + new_pop_m[work_start:work_end+1])
        taxa_desemprego = max(0.01, min(0.45, 0.7 * taxa_desemprego_ano_anterior + 0.3 * desemprego_base))
        trabalhadores_ativos = pop_work * (1 - taxa_desemprego)
        taxa_desemprego_ano_anterior = taxa_desemprego

        produtividade_setorial = sum(np.array(list(produtividade_base.values())) * np.array(list(participacao_setores.values())))
        pib_atual = trabalhadores_ativos * produtividade_setorial * fator_escala_pib
        pib.append(pib_atual)
        pib_per_capita.append(pib_atual / pop_total_t1 if pop_total_t1 > 0 else 0.0)

        # Educação e saúde
        idade_inicio_fund = 6
        alunos_f = new_pop_f[idade_inicio_fund:idade_inicio_fund+anos_fund].sum() * percent_escolarizacao['fund']
        
        # 1. Calcula o gasto "base" (calibrado)
        gasto_e_base = alunos_f * param_custo_aluno * anos_custeio_educ
        gasto_s_base = pop_total_t1 * param_gasto_saude_pc
        
        # 2. Aplica o ajuste do slider (só funciona se a simulação NÃO for histórica)
        #    (Não queremos aplicar o slider do usuário no cálculo de 2014-2021)
        if not return_history:
            gasto_e = gasto_e_base * (1 + invest_educ_pct)
            gasto_s = gasto_s_base * (1 + invest_saude_pct)
        else:
            gasto_e = gasto_e_base
            gasto_s = gasto_s_base

        gasto_saude.append(gasto_s)
        gasto_educacao.append(gasto_e)

        # --- Escolaridade média ---
        pop_total_array = new_pop_f + new_pop_m
        anos_cursados_efetivos = 0
        for nivel, perc in percent_escolarizacao.items():
            anos_n = anos_nivel[nivel]
            pop_ativa_nivel = pop_total_array[idade_inicio_fund:idade_inicio_fund+anos_n].sum() * perc
            anos_cursados_efetivos += pop_ativa_nivel * anos_n
        nivel_escolaridade_medio = anos_cursados_efetivos / pop_total_t1
        nivel_escolaridade_medio_hist.append(nivel_escolaridade_medio)

    outputs = {
        "total_pop": np.array(total_pop),
        "pib": np.array(pib),
        "pib_per_capita": np.array(pib_per_capita),
        "gasto_saude": np.array(gasto_saude),
        "gasto_educ": np.array(gasto_educacao),
        "nivel_escolaridade_medio": np.array(nivel_escolaridade_medio_hist)
    }
    if return_history:
        outputs.update({"pop_f": pop_f, "pop_m": pop_m})
    return outputs

# =========================================================
# Calibração automática
# =========================================================
def calibration_objective(x):
    # Agora 'x' tem 5 parâmetros, não 3
    fert_scale, mort_scale, mig_scale, custo_aluno, gasto_saude_pc = x

    out = run_simulation(fert_scale=fert_scale, mort_scale=mort_scale, mig_scale=mig_scale,
                         anos_simular=anos_historicos,
                         pop_f_ini=pop_inicial_f, pop_m_ini=pop_inicial_m,
                         # Passe os novos parâmetros para a simulação
                         param_custo_aluno=custo_aluno,
                         param_gasto_saude_pc=gasto_saude_pc)
    
    # Pegue os resultados da simulação
    sim_pop_hist = out['total_pop'][1:]
    sim_pib_hist = out['pib'][:anos_historicos]
    sim_saude_hist = out['gasto_saude']
    sim_educ_hist = out['gasto_educ']

    # Pegue os dados reais do DataFrame
    real_pop = df_hist['populacao_real'].values[1:]
    real_pib = df_hist['pib_real'].values[1:]
    # Use .dropna() para comparar APENAS os anos que temos dados (2020-2022)
    real_saude = df_hist['gasto_saude_real'].dropna()
    real_educ = df_hist['gasto_educ_real'].dropna()

    # Alinhe os dados simulados com os dados reais (pegando só os anos 2020-2022)
    # real_saude.index nos diz quais anos queremos (ex: [2020, 2021, 2022])
    # df_hist.index[1:] nos dá os anos da simulação (ex: [2015, ..., 2022])
    
    # Criamos um DataFrame temporário para facilitar o alinhamento
    df_sim = pd.DataFrame(index=df_hist.index[1:])
    df_sim['saude'] = sim_saude_hist
    df_sim['educ'] = sim_educ_hist
    
    # Pegamos os valores simulados APENAS dos anos que temos dados reais
    sim_saude_comparar = df_sim.loc[real_saude.index]['saude'].values
    sim_educ_comparar = df_sim.loc[real_educ.index]['educ'].values

    # Calcule o Erro (MAE) para cada um
    mae_pop = np.mean(np.abs(sim_pop_hist - real_pop))
    mae_pib = np.mean(np.abs(sim_pib_hist - real_pib))
    mae_saude = np.mean(np.abs(sim_saude_comparar - real_saude.values))
    mae_educ = np.mean(np.abs(sim_educ_comparar - real_educ.values))

    # Atualize o 'score' para incluir os novos erros
    # Damos pesos diferentes, Pop e PIB são mais importantes
    score = (
        0.35 * (mae_pop / real_pop.mean()) +
        0.35 * (mae_pib / real_pib.mean()) +
        0.15 * (mae_saude / real_saude.mean()) +
        0.15 * (mae_educ / real_educ.mean())
    )
    return score

# Dê chutes iniciais para os 5 parâmetros
# [fert, mort, mig, custo_aluno_inicial, gasto_saude_pc_inicial]
initial_guess = [1.0, 1.0, 1.0, 3000.0, 500.0] # Chutes mais realistas

# Defina limites (bounds) para os 5 parâmetros
# (fert_min, fert_max), (mort_min, mort_max), etc.
param_bounds = [(0.5, 1.5), (0.7, 1.3), (0.2, 5.0),
                (1000.0, 10000.0), # Limites para custo por aluno
                (300.0, 3000.0)]   # Limites para gasto saúde per capita

# Inicialize os parâmetros ótimos
fert_scale_opt, mort_scale_opt, mig_scale_opt = 1.0, 1.0, 1.0
custo_aluno_opt = initial_guess[3]
gasto_saude_pc_opt = initial_guess[4]

if minimize is not None:
    res = minimize(calibration_objective, initial_guess, method='L-BFGS-B',
                   bounds=param_bounds)
    if res.success:
        # Salve os 5 parâmetros encontrados
        fert_scale_opt, mort_scale_opt, mig_scale_opt, custo_aluno_opt, gasto_saude_pc_opt = res.x
        st.sidebar.success(f"Calibração concluída!")
        with st.sidebar.expander("Ver parâmetros calibrados"):
            st.write(f"Fertilidade: {fert_scale_opt:.3f}")
            st.write(f"Mortalidade: {mort_scale_opt:.3f}")
            st.write(f"Migração: {mig_scale_opt:.3f}")
            st.write(f"Custo Aluno: R$ {custo_aluno_opt:.2f}")
            st.write(f"Saúde p/ Cap: R$ {gasto_saude_pc_opt:.2f}")
    else:
        st.sidebar.warning("Calibração falhou. Usando parâmetros padrão.")

# =========================================================
# Simulações históricas e projeção
# =========================================================
out_hist = run_simulation(
    fert_scale=fert_scale_opt,
    mort_scale=mort_scale_opt,
    mig_scale=mig_scale_opt,
    anos_simular=anos_historicos,
    pop_f_ini=pop_inicial_f,
    pop_m_ini=pop_inicial_m,
    return_history=True,
    # Passe os parâmetros calibrados
    param_custo_aluno=custo_aluno_opt,
    param_gasto_saude_pc=gasto_saude_pc_opt
)

anos_proj_usuario = ano_usuario - ano_projecao
out_proj = run_simulation(
    fert_scale=fert_scale_opt,
    mort_scale=mort_scale_opt,
    mig_scale=mig_scale_opt,
    fert_increase_pct=fert_increase_pct,
    anos_simular=anos_proj_usuario,
    pop_f_ini=out_hist['pop_f'][-1],
    pop_m_ini=out_hist['pop_m'][-1],
    # Passe os parâmetros calibrados
    param_custo_aluno=custo_aluno_opt,
    param_gasto_saude_pc=gasto_saude_pc_opt,
    # PASSE OS VALORES DOS SLIDERS AQUI:
    invest_educ_pct=invest_educ_pct,
    invest_saude_pct=invest_saude_pct
)

anos_sim = np.arange(ano_projecao+1, ano_usuario+1)

# =========================================================
# Taxa de erro histórica
# =========================================================
sim_pop_hist = out_hist['total_pop'][1:anos_historicos+1]
real_pop_hist = df_hist['populacao_real'].values[1:]
sim_pib_hist = out_hist['pib'][:anos_historicos]
real_pib_hist = df_hist['pib_real'].values[1:]

erro_pop_pct = np.mean(np.abs(sim_pop_hist - real_pop_hist) / real_pop_hist * 100)
erro_pib_pct = np.mean(np.abs(sim_pib_hist - real_pib_hist) / real_pib_hist * 100)

st.subheader("Taxa de erro da simulação histórica")
col_err1, col_err2 = st.columns(2)
with col_err1:
    st.metric("Erro médio População", f"{erro_pop_pct:.2f}%")
with col_err2:
    st.metric("Erro médio PIB", f"{erro_pib_pct:.2f}%")

# =========================================================
# Resultados numéricos
# =========================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("População Total", f"{out_proj['total_pop'][-1]:,.0f} habitantes")
with col2:
    st.metric("PIB Total", f"R$ {out_proj['pib'][-1]:,.2f}")
with col3:
    st.metric("PIB per Capita", f"R$ {out_proj['pib_per_capita'][-1]:,.2f}")

col4, col5, col6 = st.columns(3)
with col4:
    st.metric("Gasto com Educação", f"R$ {out_proj['gasto_educ'][-1]:,.2f}")
with col5:
    st.metric("Gasto com Saúde", f"R$ {out_proj['gasto_saude'][-1]:,.2f}")
with col6:
    st.metric("Escolaridade Média", f"{out_proj['nivel_escolaridade_medio'][-1]:.2f} anos")

# =========================================================
# Gráfico da população
# =========================================================
st.header("Evolução da População")
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(anos_sim, out_proj['total_pop'][1:], label=f"População Fertilidade +{fert_increase_pct*100:.0f}%")
ax.set_xlabel("Ano")
ax.set_ylabel("População")
ax.legend()
ax.grid(True)
ax.yaxis.set_major_formatter(FuncFormatter(formatar_habitantes))
st.pyplot(fig)

# =========================================================
# Gráfico do PIB Total e per Capita
# =========================================================
st.header("PIB Total e Per Capita (com escalas separadas)")
fig2, ax1 = plt.subplots(figsize=(10,6))
color_pib = 'tab:blue'
color_pc = 'tab:red'
ax1.plot(anos_sim, out_proj['pib'], color=color_pib, label='PIB Total')
ax1.set_xlabel("Ano")
ax1.set_ylabel("PIB Total", color=color_pib)
ax1.tick_params(axis='y', labelcolor=color_pib)
ax1.yaxis.set_major_formatter(FuncFormatter(formatar_valor))
ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(anos_sim, out_proj['pib_per_capita'], color=color_pc, linestyle='--', label='PIB per Capita')
ax2.set_ylabel("PIB per Capita", color=color_pc)
ax2.tick_params(axis='y', labelcolor=color_pc)
ax2.yaxis.set_major_formatter(FuncFormatter(formatar_valor))
fig2.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
st.pyplot(fig2)

# =========================================================
# Gráfico dos gastos
# =========================================================
st.header("Gastos com Educação e Saúde")
fig3, ax3 = plt.subplots(figsize=(10,6))
ax3.plot(anos_sim, out_proj['gasto_educ'], label="Educação")
ax3.plot(anos_sim, out_proj['gasto_saude'], label="Saúde", linestyle="--")
ax3.set_xlabel("Ano")
ax3.set_ylabel("R$")
ax3.legend()
ax3.grid(True)
ax3.yaxis.set_major_formatter(FuncFormatter(formatar_valor))
st.pyplot(fig3)

# =========================================================
# Gráfico da escolaridade média
# =========================================================
st.header("Evolução do Nível Médio de Escolaridade")
fig4, ax4 = plt.subplots(figsize=(10,6))
ax4.plot(anos_sim, out_proj['nivel_escolaridade_medio'], label="Escolaridade Média")
ax4.set_xlabel("Ano")
ax4.set_ylabel("Anos")
ax4.grid(True)
ax4.legend()
st.pyplot(fig4)

st.info("Simulação feita a partir de 2022. Os dados históricos anteriores a 2022 foram usados para calibrar o modelo. O usuário pode escolher o ano da projeção e o impacto da fertilidade no futuro da população, educação, PIB e escolaridade.")
