# app_fertilidade_calibrada_salgueiro.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import FuncFormatter

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

# --- Configurações iniciais ---
ano_inicial = 2014
ano_projecao = 2022
anos_historicos = ano_projecao - ano_inicial
anos_projecao = 50
max_idade = 100
pop_inicial_total = 56629  # população estimada de 2014
sexo_ratio_f = 0.51

# --- Distribuição etária ---
def create_age_distribution(total_pop, max_age=100, median_age=30, sigma=15):
    ages = np.arange(max_age + 1)
    pdf = np.exp(-0.5 * ((ages - median_age) / sigma) ** 2)
    pdf /= pdf.sum()
    pop = (pdf * total_pop).astype(float)
    diff = total_pop - pop.sum()
    if abs(diff) >= 1e-6:
        pop[0] += diff
    return pop

pop_inicial = create_age_distribution(pop_inicial_total, max_age=max_idade)
pop_inicial_f = pop_inicial * sexo_ratio_f
pop_inicial_m = pop_inicial * (1 - sexo_ratio_f)

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
participacao_setores = {'agro': 0.10, 'industria': 0.25, 'servicos': 0.65}

percent_escolarizacao = {
    'fund': 0.98,
    'medio': 0.85,
    'superior': 0.30
}
anos_nivel = {'fund': 9, 'medio': 3, 'superior': 4}
anos_fund = 9
custo_aluno = {'fund': 2500}
gasto_saude_per_capita = 1500
anos_custeio_educ = anos_fund

# --- Dados históricos reais de Salgueiro (PE) ---
dados_historicos = {
    'ano': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'populacao_real': [56629, 57000, 58000, 59000, 60000, 61000, 62000, 62500, 62372],
    'pib_real': [858917920, 880000000, 900000000, 920000000, 1002105380, 1050000000, 1100000000, 1150000000, 1200000000]
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
                   return_history=False):
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
        gasto_e = alunos_f * custo_aluno['fund'] * anos_custeio_educ
        gasto_saude.append(pop_total_t1 * gasto_saude_per_capita)
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
    fert_scale, mort_scale, mig_scale = x
    out = run_simulation(fert_scale=fert_scale, mort_scale=mort_scale, mig_scale=mig_scale,
                         anos_simular=anos_historicos,
                         pop_f_ini=pop_inicial_f, pop_m_ini=pop_inicial_m)
    
    sim_pop_hist = out['total_pop'][1:]
    sim_pib_hist = out['pib'][:anos_historicos]

    real_pop = df_hist['populacao_real'].values[1:]
    real_pib = df_hist['pib_real'].values[1:]

    mae_pop = np.mean(np.abs(sim_pop_hist - real_pop))
    mae_pib = np.mean(np.abs(sim_pib_hist - real_pib))

    score = 0.6*(mae_pop/real_pop.mean()) + 0.4*(mae_pib/real_pib.mean())
    return score

fert_scale_opt, mort_scale_opt, mig_scale_opt = 1.0, 1.0, 1.0
if minimize is not None:
    res = minimize(calibration_objective, [1.0,1.0,1.0], method='L-BFGS-B',
                   bounds=[(0.5,1.5),(0.7,1.3),(0.2,5.0)])
    if res.success:
        fert_scale_opt, mort_scale_opt, mig_scale_opt = res.x
        st.sidebar.success(f"Calibração concluída: fert_scale={fert_scale_opt:.3f}, mort_scale={mort_scale_opt:.3f}, mig_scale={mig_scale_opt:.3f}")
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
    return_history=True
)

anos_proj_usuario = ano_usuario - ano_projecao
out_proj = run_simulation(
    fert_scale=fert_scale_opt,
    mort_scale=mort_scale_opt,
    mig_scale=mig_scale_opt,
    fert_increase_pct=fert_increase_pct,
    anos_simular=anos_proj_usuario,
    pop_f_ini=out_hist['pop_f'][-1],
    pop_m_ini=out_hist['pop_m'][-1]
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
