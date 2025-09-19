import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================================================
# 🔧 PARÂMETROS DO MODELO
# =========================================================
ano_inicial = 2025
anos = 50
pop_inicial_total = 120000
max_idade = 100
sexo_ratio_f = 0.51

def create_age_distribution(total_pop, max_age=100, median_age=32, sigma=18):
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

# Parâmetros Demográficos
fert_by_age = np.zeros(max_idade + 1)
fert_by_age[15:50] = np.linspace(0.01, 0.08, 35)
mort_by_age_base = np.linspace(0.001, 0.15, max_idade + 1)
mig_total_base = 250
mig_by_age_dist = np.zeros(max_idade + 1)
mig_by_age_dist[20:41] = 1
mig_by_age_dist /= mig_by_age_dist.sum()

# Parâmetros de Educação
percent_escolarizacao = {'fund': 0.98, 'medio': 0.92, 'superior': 0.75}
taxa_conclusao = {'fund_para_medio': 0.85, 'medio_para_sup': 0.25}
taxa_evasao = {'fund': 0.02, 'medio': 0.03, 'superior': 0.05}

# Parâmetros Econômicos e de Bem-Estar
work_start, work_end = 18, 64
mult_educ = {'sem_formacao': 0.8, 'fund': 1.0, 'medio': 1.3, 'superior': 1.9}
produtividade_base = {'agro': 30000, 'industria': 35000, 'servicos': 50000}
participacao_setores = {'agro': 0.10, 'industria': 0.25, 'servicos': 0.65}
desemprego_base = 0.07

# Parâmetros Financeiros
custo_aluno = {'fund': 2500, 'medio': 3500, 'superior': 5000}
gasto_saude_per_capita = 1500
taxa_imposto_pib = 0.15 # 15% do PIB vira receita pública

# =========================================================
# 🔄 INICIALIZAÇÃO E CONTROLE INTERATIVO (STREAMLIT)
# =========================================================
st.title("Simulador de Cidade Interativo Avançado")
st.write("Ajuste os parâmetros para ver o impacto no desenvolvimento da cidade.")

# Controles de Parâmetros
st.sidebar.header("Controles de Políticas")
aumento_natalidade = st.sidebar.slider("Aumento na Taxa de Natalidade (%)", 0, 100, 0) / 100
invest_educacao = st.sidebar.slider("Aumento no Investimento em Educação (%)", 0, 100, 0) / 100
capacidade_escolas_inicial = {
    'fund': 28000 * (1 + invest_educacao),
    'medio': 18000 * (1 + invest_educacao),
    'superior': 9000 * (1 + invest_educacao)
}

# =========================================================
# 🔄 INICIALIZAÇÃO DINÂMICA
# =========================================================
anos_lista = list(range(ano_inicial, ano_inicial + anos + 1))
pop_f = np.zeros((anos + 1, max_idade + 1))
pop_m = np.zeros((anos + 1, max_idade + 1))
pop_f[0, :] = pop_inicial_f
pop_m[0, :] = pop_inicial_m

# Rastreamento de Nível de Educação
pop_edu = np.zeros((anos + 1, max_idade + 1, 4)) # 4 níveis: 0-Sem, 1-Fund, 2-Med, 3-Sup
# Inicialização simples
pop_edu[0, 18:65, 0] = (pop_inicial[18:65] * 0.20)
pop_edu[0, 18:65, 1] = (pop_inicial[18:65] * 0.50)
pop_edu[0, 18:65, 2] = (pop_inicial[18:65] * 0.20)
pop_edu[0, 18:65, 3] = (pop_inicial[18:65] * 0.10)
pop_edu[0, 18:65, :] /= pop_edu[0, 18:65, :].sum(axis=1)[:, np.newaxis]
pop_edu[0, 18:65, :] *= pop_inicial[18:65][:, np.newaxis]

# Listas para armazenar os resultados
total_pop = [pop_inicial.sum()]
alunos_fund = []
alunos_medio = []
alunos_superior = []
deficit_fund = []
deficit_medio = []
deficit_superior = []
trabalhadores = []
pib = []
pib_per_capita = []
gasto_saude = []
gasto_educacao = []
receita_publica = []
balanco_orcamentario = []

# =========================================================
# 🔄 SIMULAÇÃO AVANÇADA COM INTERCONEXÕES
# =========================================================
for t in range(anos):
    # --- Passo 1: Dinâmica da População (com mortalidade e migração) ---
    popf = pop_f[t, :].copy()
    popm = pop_m[t, :].copy()
    
    # Aplica a mortalidade
    popf *= (1 - mort_by_age_base)
    popm *= (1 - mort_by_age_base)

    # Envelhece a população
    new_pop_f = np.zeros_like(popf)
    new_pop_m = np.zeros_like(popm)
    new_pop_f[1:] = popf[:-1]
    new_pop_m[1:] = popm[:-1]

    # Calcula e adiciona nascimentos (com ajuste da taxa)
    fert_modificada = fert_by_age * (1 + aumento_natalidade)
    nascimentos = np.sum(popf * fert_modificada)
    new_pop_f[0] = nascimentos * sexo_ratio_f
    new_pop_m[0] = nascimentos * (1 - sexo_ratio_f)

    # Aplica a migração (agora baseada na taxa de desemprego)
    taxa_desemprego_atual = (1 - trabalhadores[-1] / np.sum(pop_f[t, work_start:work_end+1] + pop_m[t, work_start:work_end+1])) if t > 0 else desemprego_base
    mig_total_dinamica = mig_total_base * (1 - (taxa_desemprego_atual - desemprego_base)) # migração diminui com desemprego alto
    mig_in = mig_total_dinamica * mig_by_age_dist
    new_pop_f += mig_in * sexo_ratio_f
    new_pop_m += mig_in * (1 - sexo_ratio_f)
    
    pop_f[t+1, :] = new_pop_f
    pop_m[t+1, :] = new_pop_m
    total_pop.append(new_pop_f.sum() + new_pop_m.sum())

    # --- Passo 2: Dinâmica da Educação ---
    pop_edu_atual = pop_edu[t, :, :].copy()
    new_pop_edu = np.zeros_like(pop_edu_atual)
    new_pop_edu[1:, :] = pop_edu_atual[:-1, :]
    new_pop_edu[0, 0] = nascimentos # Crianças nascem sem formação

    # Fluxo de alunos (com evasão e conclusão)
    for age in range(6, max_idade):
        # Fluxo fundamental -> médio
        if age == 15:
            concluintes = pop_edu_atual[age, 1] * taxa_conclusao['fund_para_medio']
            new_pop_edu[age+1, 2] += concluintes
            new_pop_edu[age+1, 1] += pop_edu_atual[age, 1] * (1 - taxa_conclusao['fund_para_medio'] - taxa_evasao['fund'])
            new_pop_edu[age+1, 0] += pop_edu_atual[age, 1] * taxa_evasao['fund']
        # Fluxo médio -> superior
        elif age == 18:
            concluintes = pop_edu_atual[age, 2] * taxa_conclusao['medio_para_sup']
            new_pop_edu[age+1, 3] += concluintes
            new_pop_edu[age+1, 2] += pop_edu_atual[age, 2] * (1 - taxa_conclusao['medio_para_sup'] - taxa_evasao['medio'])
            new_pop_edu[age+1, 0] += pop_edu_atual[age, 2] * taxa_evasao['medio']
        else:
            new_pop_edu[age+1, :] += pop_edu_atual[age, :]
    
    pop_edu[t+1, :, :] = new_pop_edu
    
    # Contagem de alunos
    alunos_f = np.sum((pop_f[t+1] + pop_m[t+1])[6:15]) * percent_escolarizacao['fund']
    alunos_m = np.sum((pop_f[t+1] + pop_m[t+1])[15:18]) * percent_escolarizacao['medio']
    alunos_s = np.sum((pop_f[t+1] + pop_m[t+1])[18:22]) * percent_escolarizacao['superior']
    
    alunos_fund.append(alunos_f)
    alunos_medio.append(alunos_m)
    alunos_superior.append(alunos_s)
    deficit_fund.append(max(0, alunos_f - capacidade_escolas_inicial['fund']))
    deficit_medio.append(max(0, alunos_m - capacidade_escolas_inicial['medio']))
    deficit_superior.append(max(0, alunos_s - capacidade_escolas_inicial['superior']))

    # --- Passo 3: Dinâmica Econômica (PIB e Força de Trabalho) ---
    pop_work = np.sum((pop_f[t+1] + pop_m[t+1])[work_start:work_end + 1])
    trabalhadores.append(pop_work * (1 - desemprego_base))

    # Cálculo da produtividade média ponderada pelo nível de educação
    produtividade_media = 0
    pop_work_edu = pop_edu[t+1, work_start:work_end+1, :]
    total_pop_work_edu = pop_work_edu.sum()
    if total_pop_work_edu > 0:
      prod_fund = np.sum(pop_work_edu[:, 1]) * mult_educ['fund']
      prod_medio = np.sum(pop_work_edu[:, 2]) * mult_educ['medio']
      prod_sup = np.sum(pop_work_edu[:, 3]) * mult_educ['superior']
      prod_sem = np.sum(pop_work_edu[:, 0]) * mult_educ['sem_formacao']
      produtividade_media = (prod_fund + prod_medio + prod_sup + prod_sem) / total_pop_work_edu

    pib_atual = trabalhadores[-1] * (produtividade_media * np.sum(list(produtividade_base.values()) * np.array(list(participacao_setores.values()))))
    pib.append(pib_atual)
    pib_per_capita.append(pib_atual / total_pop[-1])

    # --- Passo 4: Finanças Públicas ---
    gasto_s = total_pop[-1] * gasto_saude_per_capita
    gasto_e = (alunos_fund[-1] * custo_aluno['fund'] +
               alunos_medio[-1] * custo_aluno['medio'] +
               alunos_superior[-1] * custo_aluno['superior'])
    gasto_saude.append(gasto_s)
    gasto_educacao.append(gasto_e)

    receita = pib_atual * taxa_imposto_pib
    receita_publica.append(receita)

    balanco = receita - (gasto_s + gasto_e)
    balanco_orcamentario.append(balanco)

# =========================================================
# 🌐 INTERFACE STREAMLIT
# =========================================================
st.header("Análise dos Resultados")

# Seleção de ano e indicador
ano_usuario = st.slider("Selecione o ano para análise", ano_inicial, ano_inicial + anos -1)
idx = anos_lista.index(ano_usuario)

# Colunas para mostrar os principais indicadores
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("População Total", f"{total_pop[idx]:,.0f} habitantes")
with col2:
    st.metric("PIB Total", f"R$ {pib[idx]:,.2f}")
with col3:
    st.metric("PIB per Capita", f"R$ {pib_per_capita[idx]:,.2f}")

st.markdown("---")
st.subheader("Balanço Orçamentário Anual")
col_b1, col_b2, col_b3 = st.columns(3)
with col_b1:
    st.metric("Receita Pública", f"R$ {receita_publica[idx]:,.2f}")
with col_b2:
    st.metric("Gastos Totais", f"R$ {gasto_saude[idx] + gasto_educacao[idx]:,.2f}")
with col_b3:
    st.metric("Balanço Orçamentário", f"R$ {balanco_orcamentario[idx]:,.2f}")


# Gráficos interativos
st.header("Gráficos de Evolução")
opcoes_grafico = st.multiselect(
    "Selecione os indicadores para o gráfico",
    ["População Total", "PIB Total", "PIB per Capita", "Força de Trabalho",
     "Déficit Vagas Fund.", "Déficit Vagas Médio", "Déficit Vagas Superior",
     "Gasto com Saúde", "Gasto com Educação", "Balanço Orçamentário"],
    default=["População Total", "PIB Total", "Balanço Orçamentário"]
)

fig, ax = plt.subplots(figsize=(10, 6))

data = {
    "População Total": total_pop[:-1],
    "PIB Total": pib,
    "PIB per Capita": pib_per_capita,
    "Força de Trabalho": trabalhadores,
    "Déficit Vagas Fund.": deficit_fund,
    "Déficit Vagas Médio": deficit_medio,
    "Déficit Vagas Superior": deficit_superior,
    "Gasto com Saúde": gasto_saude,
    "Gasto com Educação": gasto_educacao,
    "Balanço Orçamentário": balanco_orcamentario,
}

for opcao in opcoes_grafico:
    ax.plot(anos_lista[:-1], data[opcao], label=opcao)

ax.set_xlabel("Ano")
ax.set_ylabel("Valor")
ax.set_title("Evolução dos Indicadores ao Longo do Tempo")
ax.legend()
ax.grid(True)
st.pyplot(fig)