# Título do Projeto

Família+IA: Sistema Nacional Inteligente de Política Demográfica
Projeto final para o curso de IA de Construção

## Resumo

Uma plataforma de inteligência artificial integrada que atua como “gestor preditivo da demografia nacional”. Utiliza dados em tempo real para otimizar incentivos à natalidade, planeamento urbano, habitação familiar, retenção de talento jovem e imigração qualificada, com foco em transformar a parentalidade em um investimento económico sustentável para famílias.

## Fundo

Portugal enfrenta baixa natalidade, envelhecimento acelerado, emigração jovem e desigualdades regionais, que ameaçam a sustentabilidade económica e social a longo prazo.
Frequência:
Taxa de fertilidade ≈ 1,4 (muito abaixo de 2,1 necessário para reposição).
Idade média do primeiro filho ≈ 30–31 anos.
Interior do país despovoado, litoral sobrecarregado.
Motivação pessoal:
Garantir sustentabilidade social e económica do país.
Tornar a parentalidade financeiramente viável e socialmente valorizada.
Usar IA para transformar políticas públicas tradicionais em soluções proativas e personalizadas.
Importância:
Evita declínio populacional catastrófico.
Garante continuidade da força de trabalho e da Segurança Social.
Cria um modelo europeu inovador de gestão demográfica inteligente.

## Tecnica de dados e IA

Fontes de dados:
Instituto Nacional de Estatística – demografia, natalidade, mortalidade
Segurança Social – dados sobre famílias, emprego, contribuições
Finanças – rendimento e impostos das famílias
Ministério da Educação – vagas escolares e creches
Habitação – preços, imóveis devolutos, programas públicos
Mobilidade e imigração – dados de entrada/saída de população
Dados socioeconómicos de base regional (municipalidades)
Técnicas de IA:
Modelos preditivos de fertilidade: regressão e redes neurais para prever nascimentos por região, idade, rendimento e políticas aplicadas.
Otimização de incentivos personalizados: algoritmos de recomendação que calculam o pacote ideal de incentivos (financeiros, fiscais, habitação) para cada família.
Planeamento urbano inteligente: IA para alocação de creches, escolas e habitação familiar.
Simulação de políticas públicas: “digital twins” demográficos para testar cenários de impacto antes da implementação.
IA de matching social: algoritmos para conectar indivíduos ou famílias, promovendo estabilidade conjugal e parentalidade planejada.
Demonstração/protótipo possível:
Base de dados fictícia de 10.000 famílias com idade, rendimento, filhos e localização.
Rede neural que prevê a probabilidade de ter um filho nos próximos 5 anos com diferentes pacotes de incentivos.
Visualização do impacto de políticas regionais em dashboards interativos.

## Como é usado?

Contexto:
Utilizado pelo Estado como ferramenta de apoio à política demográfica.
Acesso controlado por ministérios: Família, Educação, Habitação, Economia.
Usuários afetados:
Famílias e futuros pais: recebem incentivos personalizados e orientação sobre recursos públicos.
Governo e municípios: planeiam creches, escolas, habitação e políticas de imigração.
Sociedade: beneficia-se de sustentabilidade económica e demográfica.
Exemplo de fluxo de uso:
Uma família cria perfil na plataforma.
A IA calcula incentivos financeiros, habitação e serviços disponíveis.
Dashboard do município ajusta número de creches e escolas conforme a previsão de nascimentos.
Simulações de políticas ajudam o governo a decidir quais medidas regionais implementar.

## Challenges / Desafios
Privacidade de dados: dados sensíveis de famílias e rendimento precisam de proteção máxima.
Aceitação social: incentivos econômicos e matchmaking podem ser politicamente sensíveis.
Precisão preditiva: mudanças culturais ou económicas podem reduzir a eficácia do modelo.
Sustentabilidade financeira: fundos para o “Dividendo Demográfico” precisam de viabilidade a longo prazo.
Complexidade tecnológica: integração de múltiplos sistemas nacionais e atualização em tempo real.

## O que vem a seguir?

Expansão regional: iniciar piloto em duas ou três regiões com baixa natalidade.
Integração com IA avançada: uso de aprendizado por reforço para otimizar políticas em tempo real.
Gamificação e educação parental: apps para engajar famílias na gestão do Dividendo Demográfico.
Abertura a pesquisadores: permitir análises de dados anonimizados para melhorar previsões.
Escala europeia: modelo pode ser exportado para outros países enfrentando problemas semelhantes.

## Reconhecimentos

Acknowledgments / Agradecimentos
Dados do Instituto Nacional de Estatística e da Segurança Social
Inspiração: modelos familiares de França e Suécia
Pesquisa de referência: relatórios da OECD sobre fertilidade e políticas familiares
Conceitos de Dividendo Demográfico baseados em estudos demográficos internacionais

IA-para-inverter-a-crise-demografica-em-Portugal
Família+IA: Sistema Nacional Inteligente de Política Demográfica

## escreva sua solução aqui

Vamos criar um protótipo funcional em Python que demonstra a lógica central da plataforma Família+IA: previsão de natalidade e cálculo do Dividendo Demográfico para famílias, com visualização simplificada por região. Usaremos dados fictícios para simular o sistema.

## Protótipo Família+IA 1.0: Previsão de Natalidade e Dividendo Demográfico
O que este protótipo faz:
Gera dados fictícios de 1000 famílias, incluindo idade dos pais, rendimento, número de filhos e região.
Calcula uma probabilidade simplificada de ter filhos nos próximos 5 anos.
Treina um modelo de regressão logística para simular predição de natalidade com base nas variáveis familiares.
Calcula o Dividendo Demográfico anual previsto para cada família, ajustado pela probabilidade de ter filhos.
Visualiza média do Dividendo Demográfico por região, permitindo identificar zonas com maior impacto.
Lista as famílias com maior probabilidade de gerar filhos, útil para decisões de política personalizada.

# Protótipo Família+IA 1.0: Previsão de Natalidade e Dividendo Demográfico

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ----------------------------
# 1. Criar base de dados fictícia
# ----------------------------
np.random.seed(42)
n_familias = 1000

# Dados de famílias
familias = pd.DataFrame({
    'idade_mae': np.random.randint(20, 40, n_familias),
    'idade_pai': np.random.randint(22, 45, n_familias),
    'rendimento_mensal': np.random.randint(800, 5000, n_familias),
    'filhos_atuais': np.random.randint(0, 3, n_familias),
    'regiao': np.random.choice(['Lisboa', 'Porto', 'Alentejo', 'Algarve'], n_familias),
    'educacao_mae': np.random.choice([0, 1], n_familias),  # 0 = ensino médio, 1 = superior
    'estabilidade_emprego': np.random.choice([0, 1], n_familias)  # 1 = contrato fixo
})

# ----------------------------
# 2. Criar variável alvo: probabilidade de ter um filho nos próximos 5 anos
# ----------------------------
# Base simples: mães mais jovens, renda média, estabilidade de emprego, menos filhos → mais probabilidade
familias['prob_filhos'] = (
    0.3 * (40 - familias['idade_mae'])/20 +
    0.2 * familias['estabilidade_emprego'] +
    0.2 * (3 - familias['filhos_atuais'])/3 +
    0.1 * familias['educacao_mae'] +
    0.2 * (familias['rendimento_mensal'] < 3000).astype(int)
)
# Normalizar entre 0 e 1
familias['prob_filhos'] = familias['prob_filhos'].clip(0,1)

# ----------------------------
# 3. Criar modelo preditivo (simulação)
# ----------------------------
X = familias[['idade_mae','idade_pai','rendimento_mensal','filhos_atuais','educacao_mae','estabilidade_emprego']]
y = (familias['prob_filhos'] > 0.5).astype(int)

model = LogisticRegression()
model.fit(X, y)

# Predição
familias['pred_prob_filhos'] = model.predict_proba(X)[:,1]

# ----------------------------
# 4. Cálculo do Dividendo Demográfico
# ----------------------------
# Suposição de política:
# - 1º filho: 3000€/ano
# - 2º filho: 3500€/ano
# - 3º filho: 4000€/ano
def calcular_dividendo(filhos_atuais, pred_prob):
    valor_base = 3000
    bonus = filhos_atuais * 500
    dividendo = valor_base + bonus
    # Ajustar pelo risco: multiplicar pela probabilidade prevista de ter filho
    return dividendo * pred_prob

familias['dividendo_previsto'] = familias.apply(lambda row: calcular_dividendo(row['filhos_atuais'], row['pred_prob_filhos']), axis=1)

# ----------------------------
# 5. Visualização regional
# ----------------------------
dividendo_por_regiao = familias.groupby('regiao')['dividendo_previsto'].mean().sort_values()
dividendo_por_regiao.plot(kind='barh', figsize=(8,5), color='skyblue')
plt.xlabel('Dividendo Demográfico Médio Previsto (€ / ano)')
plt.ylabel('Região')
plt.title('Dividendo Demográfico Médio por Região (Protótipo Família+IA)')
plt.show()

# ----------------------------
# 6. Exemplo de famílias com maior probabilidade de ter filhos
# ----------------------------
top_familias = familias.sort_values('pred_prob_filhos', ascending=False).head(5)
print("Top 5 famílias com maior probabilidade de ter filhos nos próximos 5 anos:")
print(top_familias[['idade_mae','idade_pai','rendimento_mensal','filhos_atuais','regiao','pred_prob_filhos','dividendo_previsto']])

## Protótipo 2.0 para uma simulação interativa de políticas públicas, mantendo a base do Família+IA. A ideia é mostrar como diferentes incentivos impactam probabilidade de ter filhos e Dividendo Demográfico, permitindo decisões baseadas em dados.
O que este protótipo 2.0 faz:
Mantém a base de famílias fictícias do protótipo inicial.
Define três políticas públicas simuladas: abono familiar, creches gratuitas e habitação priorizada.
Ajusta a probabilidade de ter filhos com base nessas políticas.
Calcula o Dividendo Demográfico esperado para cada família.
Visualiza impacto médio por região, permitindo identificar onde as políticas seriam mais eficazes.
Permite comparar cenário sem políticas e cenário com políticas, mostrando o efeito das intervenções.
Identifica as famílias mais beneficiadas, útil para políticas personalizadas.

# Protótipo Família+IA 2.0: Simulação de políticas públicas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_familias = 1000

# ----------------------------
# 1. Criar base de dados fictícia
# ----------------------------
familias = pd.DataFrame({
    'idade_mae': np.random.randint(20, 40, n_familias),
    'idade_pai': np.random.randint(22, 45, n_familias),
    'rendimento_mensal': np.random.randint(800, 5000, n_familias),
    'filhos_atuais': np.random.randint(0, 3, n_familias),
    'regiao': np.random.choice(['Lisboa', 'Porto', 'Alentejo', 'Algarve'], n_familias),
    'educacao_mae': np.random.choice([0, 1], n_familias),
    'estabilidade_emprego': np.random.choice([0, 1], n_familias)
})

# ----------------------------
# 2. Definir políticas simuladas
# ----------------------------
# Estrutura: (valor anual em euros)
politicas = {
    'abono_familiar': {0: 3000, 1: 3500, 2: 4000},   # 1º, 2º, 3º filho
    'creche_gratuita': 500,  # valor estimado de economia
    'habitação_priorizada': 1000  # economia estimada para a família
}

# Multiplicadores de probabilidade de ter filhos
# Valores entre 0 e 1
politicas_prob = {
    'abono_familiar': 0.15,
    'creche_gratuita': 0.10,
    'habitação_priorizada': 0.20
}

# ----------------------------
# 3. Função de cálculo do Dividendo e probabilidade
# ----------------------------
def simular_politica(familias, politicas, politicas_prob):
    familias = familias.copy()
    
    # Probabilidade base simplificada
    familias['prob_base'] = (
        0.3 * (40 - familias['idade_mae'])/20 +
        0.2 * familias['estabilidade_emprego'] +
        0.2 * (3 - familias['filhos_atuais'])/3 +
        0.1 * familias['educacao_mae'] +
        0.2 * (familias['rendimento_mensal'] < 3000).astype(int)
    )
    familias['prob_base'] = familias['prob_base'].clip(0,1)
    
    # Ajuste pelos incentivos
    familias['prob_ajustada'] = familias['prob_base']
    # Abono familiar
    abono = familias['filhos_atuais'].apply(lambda x: politicas['abono_familiar'].get(x, 4000))
    familias['prob_ajustada'] += politicas_prob['abono_familiar'] * (abono/4000)
    # Creche gratuita
    familias['prob_ajustada'] += politicas_prob['creche_gratuita']
    # Habitação priorizada
    familias['prob_ajustada'] += politicas_prob['habitação_priorizada']
    
    familias['prob_ajustada'] = familias['prob_ajustada'].clip(0,1)
    
    # Calcular dividendo esperado
    familias['dividendo_previsto'] = (
        abono + politicas['creche_gratuita'] + politicas['habitação_priorizada']
    ) * familias['prob_ajustada']
    
    return familias

# ----------------------------
# 4. Simular efeito das políticas
# ----------------------------
familias_sim = simular_politica(familias, politicas, politicas_prob)

# ----------------------------
# 5. Visualização por região
# ----------------------------
dividendo_por_regiao = familias_sim.groupby('regiao')['dividendo_previsto'].mean().sort_values()
dividendo_por_regiao.plot(kind='barh', figsize=(8,5), color='lightgreen')
plt.xlabel('Dividendo Demográfico Médio Previsto (€ / ano)')
plt.ylabel('Região')
plt.title('Simulação do Dividendo Demográfico com Políticas Públicas')
plt.show()

# ----------------------------
# 6. Comparação antes e depois das políticas
# ----------------------------
print("Dividendo médio sem políticas (protótipo 1.0):")
print(familias['filhos_atuais'].apply(lambda x: politicas['abono_familiar'].get(x,4000)*0.5).mean())

print("\nDividendo médio com políticas aplicadas (protótipo 2.0):")
print(familias_sim['dividendo_previsto'].mean())

# ----------------------------
# 7. Top 5 famílias mais beneficiadas
# ----------------------------
top_familias = familias_sim.sort_values('dividendo_previsto', ascending=False).head(5)
print("\nTop 5 famílias com maior Dividendo Demográfico previsto após políticas:")
print(top_familias[['idade_mae','idade_pai','rendimento_mensal','filhos_atuais','regiao','prob_ajustada','dividendo_previsto']])

## Protótipo Família+IA 3.0: Versão interativa usando Python + ipywidgets, que permitirá ajustar os incentivos em tempo real e visualizar imediatamente o impacto no Dividendo Demográfico e na probabilidade de natalidade por região.
Como funciona este protótipo interativo
Você ajusta valores de abonos, creche e habitação usando sliders.
O gráfico mostra média do Dividendo Demográfico por região em tempo real.
O print abaixo do gráfico mostra o dividendo médio nacional e as 5 famílias mais beneficiadas.
Pode testar cenários diferentes, simulando políticas mais agressivas ou mais modestas.

# Protótipo Família+IA 3.0: Versão interativa, que permitirá ajustar os incentivos em tempo real e visualizar imediatamente o impacto no Dividendo Demográfico e na probabilidade de natalidade por região.

# Requer Jupyter Notebook ou JupyterLab com ipywidgets instalado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider

np.random.seed(42)
n_familias = 1000

# ----------------------------
# 1. Base de dados fictícia
# ----------------------------
familias = pd.DataFrame({
    'idade_mae': np.random.randint(20, 40, n_familias),
    'idade_pai': np.random.randint(22, 45, n_familias),
    'rendimento_mensal': np.random.randint(800, 5000, n_familias),
    'filhos_atuais': np.random.randint(0, 3, n_familias),
    'regiao': np.random.choice(['Lisboa', 'Porto', 'Alentejo', 'Algarve'], n_familias),
    'educacao_mae': np.random.choice([0, 1], n_familias),
    'estabilidade_emprego': np.random.choice([0, 1], n_familias)
})

# ----------------------------
# 2. Função de simulação interativa
# ----------------------------
def simular_interativa(abono_1=3000, abono_2=3500, abono_3=4000,
                       creche_valor=500, habitacao_valor=1000):
    familias_sim = familias.copy()
    
    # Probabilidade base
    familias_sim['prob_base'] = (
        0.3 * (40 - familias_sim['idade_mae'])/20 +
        0.2 * familias_sim['estabilidade_emprego'] +
        0.2 * (3 - familias_sim['filhos_atuais'])/3 +
        0.1 * familias_sim['educacao_mae'] +
        0.2 * (familias_sim['rendimento_mensal'] < 3000).astype(int)
    )
    familias_sim['prob_base'] = familias_sim['prob_base'].clip(0,1)
    
    # Abonos por filho
    abonos = familias_sim['filhos_atuais'].apply(lambda x: [abono_1, abono_2, abono_3][x] if x<3 else abono_3)
    
    # Ajuste probabilidade pelos incentivos
    familias_sim['prob_ajustada'] = familias_sim['prob_base'] + 0.15*(abonos/abono_3) + 0.10*(creche_valor/500) + 0.20*(habitacao_valor/1000)
    familias_sim['prob_ajustada'] = familias_sim['prob_ajustada'].clip(0,1)
    
    # Dividendo Demográfico
    familias_sim['dividendo_previsto'] = (abonos + creche_valor + habitacao_valor) * familias_sim['prob_ajustada']
    
    # Visualização
    dividendo_por_regiao = familias_sim.groupby('regiao')['dividendo_previsto'].mean().sort_values()
    plt.figure(figsize=(8,5))
    dividendo_por_regiao.plot(kind='barh', color='lightcoral')
    plt.xlabel('Dividendo Demográfico Médio Previsto (€ / ano)')
    plt.ylabel('Região')
    plt.title('Simulação Interativa do Dividendo Demográfico')
    plt.show()
    
    print(f"Dividendo médio nacional previsto: €{familias_sim['dividendo_previsto'].mean():.2f}/ano")
    top5 = familias_sim.sort_values('dividendo_previsto', ascending=False).head(5)
    print("\nTop 5 famílias mais beneficiadas:")
    display(top5[['idade_mae','idade_pai','rendimento_mensal','filhos_atuais','regiao','prob_ajustada','dividendo_previsto']])

# ----------------------------
# 3. Interface interativa
# ----------------------------
interact(
    simular_interativa,
    abono_1=IntSlider(min=1000, max=5000, step=500, value=3000, description='1º Filho'),
    abono_2=IntSlider(min=1500, max=5500, step=500, value=3500, description='2º Filho'),
    abono_3=IntSlider(min=2000, max=6000, step=500, value=4000, description='3º Filho'),
    creche_valor=IntSlider(min=0, max=2000, step=100, value=500, description='Creche'),
    habitacao_valor=IntSlider(min=0, max=3000, step=100, value=1000, description='Habitação'))

## Protótipo Família+IA 4.0: Versão final ultra-detalhada do Família+IA, que será uma ferramenta estratégica completa para Portugal, incluindo:
1. Funcionalidades principais da versão final
Mapa de calor georreferenciado por município/freguesia
Mostra o Dividendo Demográfico médio e projeção de natalidade.
Permite identificar regiões críticas ou oportunidades para políticas específicas.
Simulação multi-anos completa
Projeção de natalidade, mortalidade, imigração e emigração até 2035 ou mais.
Ajuste de políticas em tempo real para ver impacto cumulativo ao longo dos anos.
Comparação de múltiplos cenários
Cenários “sem políticas”, “políticas moderadas” e “políticas agressivas”.
Dashboards lado a lado ou gráficos interativos que mostram diferença percentual no Dividendo Demográfico.
Dashboard interativo avançado
Sliders para ajustar:
Abono familiar 1º, 2º e 3º filho
Creche gratuita
Habitação priorizada
Incentivos fiscais e sociais adicionais
Atualização imediata de mapas e gráficos.
Top famílias e agregados regionais
Tabelas interativas com famílias mais beneficiadas.
Estatísticas agregadas por município, distrito ou nacional.
Exportação de relatórios
PDF ou Excel com mapas, gráficos e tabelas por região.
Cenários comparativos documentados para decisões governamentais.

2. Arquitetura técnica sugerida
Componente	Tecnologia / Biblioteca
Frontend interativo	Dash + Plotly, Mapbox para mapas de calor
Backend / Dados	Python, Pandas, NumPy, GeoPandas
Simulações multi-anos	Funções probabilísticas de natalidade/mortalidade/fluxos migratórios
Banco de dados	PostgreSQL/PostGIS para dados georreferenciados reais
Exportação relatórios	Plotly + PDF/Excel via Pandas/ReportLab
Atualização de dados	Scripts de ingestão de dados oficiais do INE, Segurança Social e Finanças

3. Exemplo de simulação multi-anos avançada
Para cada família:
Probabilidade de ter filho ajustada por idade, rendimento, estabilidade de emprego, educação e políticas.
Probabilidade de imigrar ou emigrar baseada em região e faixa etária.
Cálculo do Dividendo Demográfico anual, acumulado por família e região.
Agregação e criação de mapas de calor para visualização estratégica.

4. Visualizações sugeridas
Mapa de calor municipal/freguesia
Cores representam Dividendo Médio Previsto.
Hover mostra: população, média de filhos, dividendo médio.
Gráfico de linha multi-cenário
Eixo X: anos (2026–2035)
Eixo Y: Dividendo Médio Nacional
Linhas: sem políticas vs políticas moderadas vs políticas agressivas
Bar plots por região
Comparação de dividendo médio por município, antes e depois das políticas
Tabelas interativas
Top 10 ou Top 50 famílias mais beneficiadas
Estatísticas regionais detalhadas: média de filhos, probabilidade de natalidade, dividendo anual

5. Escalabilidade e expansão futura
Integração com dados oficiais reais do INE e Segurança Social.
Possibilidade de adicionar IA de otimização: sugerir políticas ideais por município.
Simulações “what-if” para planeamento de longo prazo, incluindo efeitos de imigração e envelhecimento populacional.
Exportação automática de dashboards e mapas para decisão governamental.

# Protótipo Final Família+IA: Dashboard Estratégico Multi-Anos com Mapas de Calor
# Instalar bibliotecas: pip install dash pandas numpy plotly geopandas

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd

# ----------------------------
# 1. Dados fictícios por município
# ----------------------------
np.random.seed(42)
municipios = ['Lisboa', 'Porto', 'Coimbra', 'Faro', 'Évora', 'Braga', 'Viseu', 'Aveiro']
n_familias = 5000

familias = pd.DataFrame({
    'idade_mae': np.random.randint(20,40, n_familias),
    'idade_pai': np.random.randint(22,45, n_familias),
    'rendimento_mensal': np.random.randint(800,5000, n_familias),
    'filhos_atuais': np.random.randint(0,3, n_familias),
    'municipio': np.random.choice(municipios, n_familias),
    'educacao_mae': np.random.choice([0,1], n_familias),
    'estabilidade_emprego': np.random.choice([0,1], n_familias)
})

# ----------------------------
# 2. Função de simulação multi-anos
# ----------------------------
def simular_multi_anos(familias, abonos, creche_valor, habitacao_valor, anos=15):
    df = familias.copy()
    historico = []
    
    for ano in range(1, anos+1):
        # Probabilidade base
        df['prob_base'] = (
            0.3 * (40 - df['idade_mae'])/20 +
            0.2 * df['estabilidade_emprego'] +
            0.2 * (3 - df['filhos_atuais'])/3 +
            0.1 * df['educacao_mae'] +
            0.2 * (df['rendimento_mensal'] < 3000).astype(int)
        ).clip(0,1)
        
        # Abonos por filho
        abonos_filho = df['filhos_atuais'].apply(lambda x: abonos[x] if x<3 else abonos[2])
        
        # Probabilidade ajustada
        df['prob_ajustada'] = df['prob_base'] + 0.15*(abonos_filho/abonos[2]) + 0.10*(creche_valor/500) + 0.20*(habitacao_valor/1000)
        df['prob_ajustada'] = df['prob_ajustada'].clip(0,1)
        
        # Dividendo Demográfico
        df['dividendo_previsto'] = (abonos_filho + creche_valor + habitacao_valor) * df['prob_ajustada']
        
        # Simulação de nascimento de novo filho
        nascimentos = np.random.binomial(1, df['prob_ajustada'])
        df['filhos_atuais'] += nascimentos
        
        # Agregar por município
        df_agg = df.groupby('municipio')['dividendo_previsto'].mean().reset_index()
        df_agg['ano'] = ano
        historico.append(df_agg)
        
    return pd.concat(historico, ignore_index=True)

# ----------------------------
# 3. Dash App
# ----------------------------
app = dash.Dash(__name__)
app.title = "Família+IA Estratégico"

app.layout = html.Div([
    html.H1("Família+IA Estratégico: Dashboard Multi-Anos e Mapas de Calor", style={'textAlign':'center'}),
    
    html.Div([
        html.Label("Abono 1º Filho (€)"),
        dcc.Slider(1000,6000, step=500, value=3000, id='abono1'),
        html.Label("Abono 2º Filho (€)"),
        dcc.Slider(1500,6500, step=500, value=3500, id='abono2'),
        html.Label("Abono 3º Filho (€)"),
        dcc.Slider(2000,7000, step=500, value=4000, id='abono3'),
        html.Label("Creche Gratuita (€)"),
        dcc.Slider(0,2000, step=100, value=500, id='creche'),
        html.Label("Habitação Priorizada (€)"),
        dcc.Slider(0,3000, step=100, value=1000, id='habitacao'),
        html.Label("Anos de Simulação"),
        dcc.Slider(5,30, step=1, value=15, id='anos')
    ], style={'width':'50%','margin':'auto'}),
    
    html.Hr(),
    
    dcc.Graph(id='mapa_dividendo'),
    
    html.H2("Dividendo Médio Nacional ao Longo dos Anos", style={'textAlign':'center'}),
    dcc.Graph(id='grafico_nacional')
])

# ----------------------------
# 4. Callback interativo
# ----------------------------
@app.callback(
    Output('mapa_dividendo','figure'),
    Output('grafico_nacional','figure'),
    Input('abono1','value'),
    Input('abono2','value'),
    Input('abono3','value'),
    Input('creche','value'),
    Input('habitacao','value'),
    Input('anos','value')
)
def atualizar_dashboard(ab1, ab2, ab3, creche_valor, habitacao_valor, anos):
    abonos = [ab1, ab2, ab3]
    df_sim = simular_multi_anos(familias, abonos, creche_valor, habitacao_valor, anos)
    
    # Mapa interativo por município (proxy de mapa de calor)
    fig_map = px.choropleth(
        df_sim[df_sim['ano']==anos],
        locations='municipio',
        locationmode='USA-states', # placeholder, no caso real substituir por shapefile Portugal
        color='dividendo_previsto',
        color_continuous_scale='YlOrRd',
        labels={'dividendo_previsto':'Dividendo Médio (€)'},
        title=f'Dividendo Demográfico Médio por Município no Ano {anos}'
    )
    
    # Evolução nacional
    df_nacional = df_sim.groupby('ano')['dividendo_previsto'].mean().reset_index()
    fig_nacional = px.line(df_nacional, x='ano', y='dividendo_previsto',
                           labels={'ano':'Ano','dividendo_previsto':'Dividendo Médio Nacional (€)'},
                           title='Evolução do Dividendo Demográfico Nacional')
    
    return fig_map, fig_nacional

# ----------------------------
# 5. Rodar servidor Dash
# ----------------------------
if __name__ == '__main__':
    app.run_server(debug=True)


<!-- Este é o modelo de markdown para o projeto final do curso Building AI,
criado pela Reaktor Innovations e pela Universidade de Helsínquia! -->
