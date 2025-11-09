import streamlit as st
import numpy as np
import pandas as pd

# --- Configurações Iniciais ---
st.set_page_config(layout="wide", page_title="Calculadora de Odd Value para Futebol Virtual")

# --- Constantes do Modelo Heurístico (Ajuste a seu critério) ---
# Fatores de impacto nos Gols Esperados (Expected Goals - EG) para um único evento
PESOS = {
    'CHUTE_GOL': 0.15,          # Alto impacto
    'ATAQUE_PERIGOSO': 0.05,    # Impacto médio
    'ESCANTEIO': 0.02           # Baixo impacto
}
# Linha de Gols para Análise de Over/Under
LINHA_GOL = 2.5
MAX_MINUTO = 11 # Novo limite de simulação

# --- Funções do Modelo ---

def inicializar_estado():
    """Inicializa as variáveis de estado da sessão do Streamlit."""
    if 'jogo_iniciado' not in st.session_state:
        st.session_state.jogo_iniciado = False
        st.session_state.prob_casa_inicial = 0.0
        st.session_state.prob_empate_inicial = 0.0
        st.session_state.prob_fora_inicial = 0.0
        st.session_state.gols_casa = 0
        st.session_state.gols_fora = 0
        st.session_state.minuto_atual = 0  # Começa no minuto 0 (Pré-jogo)
        st.session_state.eventos_registrados = []
        st.session_state.eg_casa = 0.0
        st.session_state.eg_fora = 0.0
        st.session_state.eg_total = 0.0
        st.session_state.simulacao_rodada = False # Novo flag para simulação
        
        # Inicializa dados da simulação (Minuto 1 a 11)
        st.session_state.simulacao_dados = pd.DataFrame({
            'Minuto': list(range(1, MAX_MINUTO + 1)),
            'Chutes a Gol': [0] * MAX_MINUTO,
            'Ataques Perigosos': [0] * MAX_MINUTO,
            'Escanteios': [0] * MAX_MINUTO,
            'Gols Casa': [0] * MAX_MINUTO,
            'Gols Fora': [0] * MAX_MINUTO
        }).set_index('Minuto')


def calcular_probabilidade_inicial(odd):
    """Converte odd para probabilidade implícita."""
    if odd > 1.0:
        return 1 / odd
    return 0.0

def calcular_eg_base(prob_win):
    """Calcula um EG base inicial a partir da probabilidade de vitória (Simplificação)."""
    # Se a probabilidade de vitória for 0.5 (odd 2.0), EG_Base será 1.0.
    return 2.0 * prob_win

def calcular_odd_justa_over(eg_total, minuto_atual, linha=LINHA_GOL):
    """
    Calcula a probabilidade do Over X.5 gols usando uma simplificação do modelo de Poisson.
    Ajusta pelo tempo restante.
    """
    # 1. Ajuste do EG pelo tempo restante (90 minutos é o padrão total do jogo, mesmo que a simulação pare em 11)
    # Se o minuto for 0, usamos o fator 1.0.
    if minuto_atual == 0:
        tempo_restante_fator = 1.0
    else:
        # A simulação vai até 11, mas a tendência se baseia no tempo total (90)
        tempo_restante_fator = (90 - minuto_atual) / 90.0
        
    eg_total_ajustado = eg_total * tempo_restante_fator
    
    # 2. Heurística para Over 2.5 (probabilidade aumenta com o EG Total)
    if linha == 2.5:
        # Fórmula sigmóide simples para transformar EG em probabilidade Over
        # O EG de 2.5 gols é o ponto onde P(Over) deve ser ~50%
        prob_over = 1 / (1 + np.exp(-(eg_total_ajustado - 2.5) * 1.5))
        prob_over = np.clip(prob_over, 0.05, 0.95)
    else: # Adaptação para outras linhas ou um modelo mais complexo
        prob_over = 1 / (1 + np.exp(-(eg_total_ajustado - linha) * 1.5))
        prob_over = np.clip(prob_over, 0.05, 0.95)

    if prob_over == 0:
        return 0, 0
        
    odd_justa_over = 1 / prob_over
    return odd_justa_over, prob_over

# --- Funções de Ação ---

def iniciar_jogo(odd_casa, odd_empate, odd_fora):
    """Calcula probabilidades iniciais e EG base e reseta estados."""
    
    # Validação de Odds
    if odd_casa <= 1.0 or odd_empate <= 1.0 or odd_fora <= 1.0:
        st.error("Odds devem ser maiores que 1.0.")
        return
        
    # Resetar estados para iniciar um novo jogo
    st.session_state.gols_casa = 0
    st.session_state.gols_fora = 0
    st.session_state.minuto_atual = 0
    st.session_state.eventos_registrados = []
    st.session_state.simulacao_rodada = False

    # Cálculo das Probabilidades
    p_casa = calcular_probabilidade_inicial(odd_casa)
    p_empate = calcular_probabilidade_inicial(odd_empate)
    p_fora = calcular_probabilidade_inicial(odd_fora)
    
    soma_p = p_casa + p_empate + p_fora
    if soma_p > 1.1 or soma_p < 0.9:
         st.warning(f"Soma das probabilidades: {soma_p:.2f}. Margem alta ou baixa demais. Os cálculos de EG base serão aproximados.")

    # Normaliza (para desconsiderar a margem da casa de apostas)
    p_casa_norm = p_casa / soma_p
    p_fora_norm = p_fora / soma_p

    # Atribui EG Base inicial (Simplificação Crua)
    eg_casa_base = calcular_eg_base(p_casa_norm)
    eg_fora_base = calcular_eg_base(p_fora_norm)
    
    st.session_state.prob_casa_inicial = p_casa_norm
    st.session_state.prob_fora_inicial = p_fora_norm
    
    # Define o EG total inicial como a soma dos EGs base
    st.session_state.eg_casa = eg_casa_base
    st.session_state.eg_fora = eg_fora_base
    st.session_state.eg_total = eg_casa_base + eg_fora_base
    
    st.session_state.jogo_iniciado = True
    st.session_state.minuto_atual = 0 # Início no Minuto 0


def registrar_evento(minuto, chutes, ataques, escanteios, gols_casa, gols_fora):
    """Processa eventos de um minuto e atualiza o estado do jogo."""
    
    # 1. Cálculo do EG adicional (Heurística)
    eg_adicional_casa = 0
    eg_adicional_fora = 0
    
    # Distribui o impacto pela Força Relativa (EG atual)
    if st.session_state.eg_total > 0:
        fator_casa = st.session_state.eg_casa / st.session_state.eg_total
        fator_fora = st.session_state.eg_fora / st.session_state.eg_total
    else:
        # Se EG Total for zero (só acontece se EG base for zero, o que é improvável), distribui 50/50
        fator_casa = 0.5
        fator_fora = 0.5

    # Aplica pesos dos eventos
    impacto_total = (chutes * PESOS['CHUTE_GOL']) + \
                    (ataques * PESOS['ATAQUE_PERIGOSO']) + \
                    (escanteios * PESOS['ESCANTEIO'])

    eg_adicional_casa = impacto_total * fator_casa
    eg_adicional_fora = impacto_total * fator_fora

    # 2. Atualiza Gols e Minuto
    st.session_state.gols_casa += gols_casa
    st.session_state.gols_fora += gols_fora
    st.session_state.minuto_atual = minuto

    # 3. Atualiza EGs Totais (acumulativos)
    # Nota: EG dos gols marcados *não* é subtraído, pois o modelo EG já representa a tendência de ataque/defesa.
    st.session_state.eg_casa += eg_adicional_casa
    st.session_state.eg_fora += eg_adicional_fora
    st.session_state.eg_total = st.session_state.eg_casa + st.session_state.eg_fora
    
    # 4. Registra e exibe evento
    # Recalcula a odd justa para este minuto para registro no histórico
    odd_minuto, prob_minuto = calcular_odd_justa_over(st.session_state.eg_total, minuto)
    
    st.session_state.eventos_registrados.append({
        'Minuto': minuto,
        'Placar': f"{st.session_state.gols_casa} x {st.session_state.gols_fora}",
        'Chutes': chutes,
        'Ataques': ataques,
        'Escanteios': escanteios,
        'EG Acumulado': f"{st.session_state.eg_total:.2f}",
        'Odd Justa': f"{odd_minuto:.2f}"
    })


def rodar_simulacao(df_eventos):
    """Roda o jogo do Minuto 1 ao 11 com base nos dados de entrada."""
    
    # Restaura EG Base
    eg_casa_base_start = st.session_state.eg_casa 
    eg_fora_base_start = st.session_state.eg_fora
    
    # Resetar estados que serão acumulados
    st.session_state.gols_casa = 0
    st.session_state.gols_fora = 0
    st.session_state.eventos_registrados = []
    st.session_state.minuto_atual = 0
    
    # Reinicia o EG para o valor base inicial
    st.session_state.eg_casa = eg_casa_base_start
    st.session_state.eg_fora = eg_fora_base_start
    st.session_state.eg_total = eg_casa_base_start + eg_fora_base_start
    
    for minuto in range(1, MAX_MINUTO + 1):
        try:
            # Obtém os dados do minuto na tabela
            dados = df_eventos.loc[minuto]
            
            # Chama a função principal de registro de evento (que atualiza o estado)
            registrar_evento(
                minuto=minuto,
                chutes=dados['Chutes a Gol'],
                ataques=dados['Ataques Perigosas'], # Note a correção no nome da coluna
                escanteios=dados['Escanteios'],
                gols_casa=dados['Gols Casa'],
                gols_fora=dados['Gols Fora']
            )
            
        except KeyError:
            st.warning(f"Dados faltando para o Minuto {minuto}. Simulação interrompida.")
            break

    st.session_state.simulacao_rodada = True
    st.toast(f"Simulação completa até o Minuto {MAX_MINUTO}!", icon="🎉")


# --- Inicialização e Layout ---

inicializar_estado()

st.title("⚽ Calculadora Odd Value Dinâmico (Futebol Virtual)")
st.caption(f"Simulação automática de tendência até o Minuto {MAX_MINUTO}. Início no Minuto 0.")

# --- Seção 1: Configuração Inicial (Sidebar) ---
with st.sidebar:
    st.header("Configuração Inicial (Pré-jogo)")
    
    col1_s, col2_s, col3_s = st.columns(3)
    odd_casa = col1_s.number_input("Odd Inicial (Casa)", min_value=1.01, value=2.20, step=0.01, key="odd_c")
    odd_empate = col2_s.number_input("Odd Inicial (Empate)", min_value=1.01, value=3.20, step=0.01, key="odd_e")
    odd_fora = col3_s.number_input("Odd Inicial (Fora)", min_value=1.01, value=3.20, step=0.01, key="odd_f")
    
    if st.button("▶️ Iniciar Jogo e Calcular Força Base", key="btn_iniciar"):
        iniciar_jogo(odd_casa, odd_empate, odd_fora)

# --- Seção 2: Jogo em Andamento ---

if st.session_state.jogo_iniciado:
    
    st.header(f"Minuto de Análise: {st.session_state.minuto_atual}")
    
    # Display Força Base
    st.markdown(
        f"""
        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
            <h4 style='margin-top: 0; font-size: 1em;'>Força Base Inicial (Pré-jogo)</h4>
            <p style='margin: 0;'>Casa: **{st.session_state.prob_casa_inicial * 100:.1f}%** | Fora: **{st.session_state.prob_fora_inicial * 100:.1f}%**</p>
            <p style='margin: 0;'>EG Base Total: **{st.session_state.eg_total:.2f}**</p>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.subheader(f"1. Dados da Simulação (Minuto 1 ao {MAX_MINUTO})")
    st.info("Ajuste os eventos (Chutes, Gols, etc.) minuto a minuto na tabela abaixo.")

    # Edição do DataFrame (Tabela de Eventos)
    edited_df = st.data_editor(
        st.session_state.simulacao_dados,
        column_config={
            "Minuto": st.column_config.NumberColumn(
                "Minuto",
                help="Minuto do Jogo",
                disabled=True
            ),
            "Chutes a Gol": st.column_config.NumberColumn(
                "Chutes a Gol",
                help="Chutes no gol naquele minuto.",
                min_value=0,
                max_value=5,
                step=1,
            ),
            "Ataques Perigosos": st.column_config.NumberColumn(
                "Ataques Perigosas", # Renomeado para refletir o nome da coluna no DataFrame
                help="Volume de jogo naquele minuto (Opcional).",
                min_value=0,
                max_value=10,
                step=1,
            ),
             "Escanteios": st.column_config.NumberColumn(
                "Escanteios",
                help="Escanteios naquele minuto (Opcional).",
                min_value=0,
                max_value=3,
                step=1,
            ),
            "Gols Casa": st.column_config.NumberColumn(
                "Gols Casa",
                help="Gols marcados pelo time da casa.",
                min_value=0,
                max_value=3,
                step=1,
            ),
            "Gols Fora": st.column_config.NumberColumn(
                "Gols Fora",
                help="Gols marcados pelo time de fora.",
                min_value=0,
                max_value=3,
                step=1,
            ),
        },
        hide_index=False,
        num_rows="dynamic",
    )
    
    # Atualiza o estado da simulação com os dados editados
    st.session_state.simulacao_dados = edited_df

    if st.button(f"🚀 Rodar Simulação até Minuto {MAX_MINUTO}"):
        rodar_simulacao(edited_df)

    st.markdown("---")
    st.subheader(f"2. Resultado Final (Após Minuto {MAX_MINUTO})")

    if st.session_state.simulacao_rodada:
        
        col_score, col_eg, col_odds = st.columns(3)
        
        col_score.metric(
            label=f"PLACAR FINAL (Min {MAX_MINUTO})",
            value=f"{st.session_state.gols_casa} x {st.session_state.gols_fora}"
        )
        
        col_eg.metric(
            label="EG Total ACUMULADO",
            value=f"{st.session_state.eg_total:.2f} Gols"
        )
        
        # Última Odd Justa Calculada (no Minuto 11)
        odd_justa, prob_justa = calcular_odd_justa_over(st.session_state.eg_total, MAX_MINUTO)
        
        if odd_justa > 0:
            
            # Input da Odd da Casa de Apostas (para o Minuto 11)
            live_odd_aposta = col_odds.number_input(
                f"Odd da Casa (Over {LINHA_GOL}) no Min {MAX_MINUTO}", 
                min_value=1.01, 
                value=2.0, 
                step=0.01,
                key="live_odd_input_final"
            )

            st.markdown("---")
            
            col_justa, col_prob, col_value = st.columns(3)
            
            # Cálculo e Exibição do Valor (Value)
            value_ratio = live_odd_aposta / odd_justa
            
            col_justa.metric(
                label=f"Odd Justa (Fair Odd) Over {LINHA_GOL}",
                value=f"{odd_justa:.2f}"
            )
            
            col_prob.metric(
                label=f"Probabilidade Justa (%)",
                value=f"{prob_justa * 100:.1f}%"
            )
            
            # Interpretação do Odd Value
            if value_ratio > 1.05:
                value_status = "✅ Forte Valor! (Value)"
                color = "#10B981" # Green
            elif value_ratio > 1.01:
                value_status = "🟡 Pequeno Valor Encontrado"
                color = "#F59E0B" # Yellow/Orange
            else:
                value_status = "❌ Sem Valor (No Value)"
                color = "#EF4444" # Red
                
            col_value.markdown(
                f"""
                <div style='background-color: {color}; padding: 15px; border-radius: 8px; color: white; text-align: center;'>
                    <h4 style='margin-top: 0; font-size: 1.2em;'>Odd Value Ratio</h4>
                    <p style='font-size: 2em; margin: 0;'>**{value_ratio:.2f}**</p>
                    <p style='margin: 0; font-weight: bold;'>{value_status}</p>
                </div>
                """, unsafe_allow_html=True
            )

            st.markdown("---")
            st.subheader("3. Histórico Detalhado (Minuto a Minuto)")
            # Exibir a tabela de histórico de eventos
            df_historico = pd.DataFrame(st.session_state.eventos_registrados)
            # Reorganiza as colunas para melhor visualização
            st.dataframe(
                df_historico[['Minuto', 'Placar', 'Chutes', 'Ataques', 'Escanteios', 'EG Acumulado', 'Odd Justa']], 
                use_container_width=True
            )

        else:
            st.warning("Não foi possível calcular o Odd Justo (EG total é 0).")
            
    else:
        st.info(f"O Placar, EG Total e Análise de Value serão exibidos após rodar a simulação até o Minuto {MAX_MINUTO}.")
        
else:
    st.info("Aguardando a inserção das odds iniciais para começar a simulação.")
