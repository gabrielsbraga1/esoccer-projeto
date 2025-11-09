import streamlit as st
import numpy as np
import pandas as pd

# --- Configurações Iniciais ---
st.set_page_config(layout="wide", page_title="Calculadora de Odd Value para Futebol Virtual")

# --- Constantes do Modelo Heurístico (Ajuste a seu critério) ---
# Fatores de impacto nos Gols Esperados (Expected Goals - EG) para um único evento
PESOS = {
    'CHUTE_GOL': 0.15,          # Alto impacto, agora atribuído diretamente ao time
    'ATAQUE_PERIGOSO': 0.05,    # Impacto médio (distribuído pela força inicial)
    'ESCANTEIO': 0.02           # Baixo impacto (distribuído pela força inicial)
}
# Linha de Gols para Análise de Over/Under
LINHA_GOL = 2.5

# --- Funções do Modelo ---

def inicializar_estado():
    """Inicializa as variáveis de estado da sessão do Streamlit."""
    if 'jogo_iniciado' not in st.session_state:
        st.session_state.jogo_iniciado = False
        st.session_state.prob_casa_inicial = 0.0
        st.session_state.prob_fora_inicial = 0.0
        st.session_state.gols_casa = 0
        st.session_state.gols_fora = 0
        st.session_state.minuto_atual = 0  # Começa no minuto 0 (Pré-jogo)
        st.session_state.eventos_registrados = []
        st.session_state.eg_casa = 0.0
        st.session_state.eg_fora = 0.0
        st.session_state.eg_total = 0.0
        st.session_state.odd_live_aposta = 2.0 # Valor inicial para aposta live


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
    if minuto_atual == 0:
        tempo_restante_fator = 1.0
    else:
        # Fator de ajuste baseado no tempo total de 90 minutos
        tempo_restante_fator = (90 - minuto_atual) / 90.0
        
    eg_total_ajustado = eg_total * tempo_restante_fator
    
    # Heurística para Over 2.5
    if linha == 2.5:
        # Fórmula sigmóide simples para transformar EG em probabilidade Over
        prob_over = 1 / (1 + np.exp(-(eg_total_ajustado - 2.5) * 1.5))
        prob_over = np.clip(prob_over, 0.05, 0.95)
    else: 
        prob_over = 1 / (1 + np.exp(-(eg_total_ajustado - linha) * 1.5))
        prob_over = np.clip(prob_over, 0.05, 0.95)

    if prob_over == 0:
        return 0, 0
        
    odd_justa_over = 1 / prob_over
    return odd_justa_over, prob_over

# --- Funções de Ação ---

def iniciar_jogo(odd_casa, odd_empate, odd_fora):
    """Calcula probabilidades iniciais e EG base e reseta estados."""
    
    if odd_casa <= 1.0 or odd_empate <= 1.0 or odd_fora <= 1.0:
        st.error("Odds devem ser maiores que 1.0.")
        return
        
    # Resetar estados para iniciar um novo jogo
    st.session_state.gols_casa = 0
    st.session_state.gols_fora = 0
    st.session_state.minuto_atual = 0
    st.session_state.eventos_registrados = []

    # Cálculo das Probabilidades
    p_casa = calcular_probabilidade_inicial(odd_casa)
    p_fora = calcular_probabilidade_inicial(odd_fora)
    
    soma_p = p_casa + calcular_probabilidade_inicial(odd_empate) + p_fora
    
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
    st.session_state.minuto_atual = 1 # Inicia o jogo no Minuto 1

def registrar_evento(minuto, chutes_casa, chutes_fora, ataques, escanteios, gols_casa, gols_fora):
    """Processa eventos de um minuto e atualiza o estado do jogo."""
    
    if minuto <= st.session_state.minuto_atual:
        st.error(f"O minuto deve ser superior ao minuto atual ({st.session_state.minuto_atual - 1}).")
        return

    # Distribui o impacto pela Força Relativa (EG atual) para eventos genéricos
    if st.session_state.eg_total > 0:
        fator_casa = st.session_state.eg_casa / st.session_state.eg_total
        fator_fora = st.session_state.eg_fora / st.session_state.eg_total
    else:
        fator_casa = 0.5
        fator_fora = 0.5

    # 1. Cálculo do EG adicional (Heurística)
    
    # EG específico dos Chutes (diretamente atribuído ao time)
    eg_chute_casa = chutes_casa * PESOS['CHUTE_GOL']
    eg_chute_fora = chutes_fora * PESOS['CHUTE_GOL']

    # Impacto Genérico (Ataques Perigosos e Escanteios, distribuído pela força relativa)
    impacto_generico = (ataques * PESOS['ATAQUE_PERIGOSO']) + \
                       (escanteios * PESOS['ESCANTEIO'])

    eg_adicional_casa = eg_chute_casa + (impacto_generico * fator_casa)
    eg_adicional_fora = eg_chute_fora + (impacto_generico * fator_fora)

    # 2. Atualiza Gols e Minuto
    st.session_state.gols_casa += gols_casa
    st.session_state.gols_fora += gols_fora
    st.session_state.minuto_atual = minuto + 1 # Próximo minuto a ser analisado

    # 3. Atualiza EGs Totais (acumulativos)
    st.session_state.eg_casa += eg_adicional_casa
    st.session_state.eg_fora += eg_adicional_fora
    st.session_state.eg_total = st.session_state.eg_casa + st.session_state.eg_fora
    
    # 4. Registra no Histórico
    odd_minuto, prob_minuto = calcular_odd_justa_over(st.session_state.eg_total, minuto)
    
    st.session_state.eventos_registrados.append({
        'Minuto': minuto,
        'Placar': f"{st.session_state.gols_casa} x {st.session_state.gols_fora}",
        'Cht C/F': f"{chutes_casa}/{chutes_fora}",
        'Atq': ataques,
        'Esc': escanteios,
        'EG Acumulado': f"{st.session_state.eg_total:.2f}",
        'Odd Justa': f"{odd_minuto:.2f}"
    })
    
    st.toast("Evento registrado e Odds recalculadas!", icon="⚽")


# --- Inicialização e Layout ---

inicializar_estado()

st.title("⚽ Odd Value Dinâmico (Futebol Virtual)")
st.caption("Calculadora em Tempo Real: Recalcula a tendência minuto a minuto.")

# --- Seção 1: Configuração Inicial (Sidebar) ---
with st.sidebar:
    st.header("Configuração Inicial (Pré-jogo)")
    
    if st.session_state.jogo_iniciado:
        st.success("Jogo Iniciado!")
    
    col1_s, col2_s, col3_s = st.columns(3)
    odd_casa = col1_s.number_input("Odd Inicial (Casa)", min_value=1.01, value=2.20, step=0.01, key="odd_c")
    odd_empate = col2_s.number_input("Odd Inicial (Empate)", min_value=1.01, value=3.20, step=0.01, key="odd_e")
    odd_fora = col3_s.number_input("Odd Inicial (Fora)", min_value=1.01, value=3.20, step=0.01, key="odd_f")
    
    if st.button("▶️ Iniciar Jogo e Calcular Força Base", key="btn_iniciar"):
        iniciar_jogo(odd_casa, odd_empate, odd_fora)

# --- Seção 2: Jogo em Andamento ---

if st.session_state.jogo_iniciado:
    
    minuto_jogo = st.session_state.minuto_atual - 1
    
    st.header(f"Minuto de Análise: {minuto_jogo}")
    
    # --- Métricas de Resumo ---
    col_score, col_eg, col_prob = st.columns(3)
    
    col_score.metric(
        label="Placar (Casa x Fora)",
        value=f"{st.session_state.gols_casa} x {st.session_state.gols_fora}"
    )
    
    col_eg.metric(
        label="EG Total Acumulado (Expected Goals)",
        value=f"{st.session_state.eg_total:.2f} Gols"
    )
    
    odd_justa_atual, prob_justa_atual = calcular_odd_justa_over(st.session_state.eg_total, minuto_jogo)
    
    col_prob.metric(
        label=f"Prob. Justa Over {LINHA_GOL} (%)",
        value=f"{prob_justa_atual * 100:.1f}%"
    )

    st.subheader(f"Registro de Eventos no Minuto {st.session_state.minuto_atual}")
    
    with st.form("registro_eventos"):
        
        # Minuto e Odd Live
        col_m, col_odd_live = st.columns(2)
        minuto_novo = col_m.number_input(
            f"Minuto a ser Registrado (Atual: {st.session_state.minuto_atual})", 
            min_value=st.session_state.minuto_atual, 
            max_value=90, 
            value=st.session_state.minuto_atual
        )
        
        live_odd_aposta = col_odd_live.number_input(
            f"Odd Live da Casa (Over {LINHA_GOL} Gols)", 
            min_value=1.01, 
            value=st.session_state.odd_live_aposta, 
            step=0.01,
            key="live_odd_input_form"
        )
        # Atualiza a odd live para o próximo ciclo
        st.session_state.odd_live_aposta = live_odd_aposta
        
        st.markdown("---")
        st.markdown("#### Eventos de Ataque")
        col_cht_c, col_cht_f, col_atq, col_esc = st.columns(4)
        
        chutes_casa = col_cht_c.number_input("Chutes a Gol (Casa)", min_value=0, value=0, step=1)
        chutes_fora = col_cht_f.number_input("Chutes a Gol (Visitante)", min_value=0, value=0, step=1)
        ataques_perigosos = col_atq.number_input("Ataques Perigosos", min_value=0, value=0, step=1)
        escanteios = col_esc.number_input("Escanteios", min_value=0, value=0, step=1)
        
        st.markdown("#### Gols Marcados")
        col_gol_casa, col_gol_fora = st.columns(2)
        
        gols_casa = col_gol_casa.number_input("Gols do Time Casa (+)", min_value=0, value=0, step=1)
        gols_fora = col_gol_fora.number_input("Gols do Time Fora (+)", min_value=0, value=0, step=1)
        
        if st.form_submit_button("🔁 Recalcular Tendência (Registrar Evento)"):
            registrar_evento(minuto_novo, chutes_casa, chutes_fora, ataques_perigosos, escanteios, gols_casa, gols_fora)

    st.markdown("---")
    st.subheader(f"📊 Análise de Odd Value (Over {LINHA_GOL}) no Minuto {minuto_jogo}")
    
    if odd_justa_atual > 0:
        
        # Cálculo do Valor (Value)
        value_ratio = live_odd_aposta / odd_justa_atual
        
        col_justa, col_aposta, col_value = st.columns(3)
        
        col_justa.metric(
            label=f"Odd Justa (Fair Odd) Over {LINHA_GOL}",
            value=f"{odd_justa_atual:.2f}"
        )

        col_aposta.metric(
            label=f"Sua Odd de Aposta",
            value=f"{live_odd_aposta:.2f}"
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
            <div style='background-color: {color}; padding: 15px; border-radius: 8px; color: white; text-align: center; height: 100%'>
                <h4 style='margin-top: 0; font-size: 1.2em;'>Odd Value Ratio</h4>
                <p style='font-size: 2em; margin: 0;'>**{value_ratio:.2f}**</p>
                <p style='margin: 0; font-weight: bold;'>{value_status}</p>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("Histórico Detalhado (Eventos Registrados)")
        if st.session_state.eventos_registrados:
            # Exibir a tabela de histórico de eventos
            df_historico = pd.DataFrame(st.session_state.eventos_registrados)
            st.dataframe(df_historico, use_container_width=True)
        else:
            st.info("Nenhum evento registrado ainda.")
            
    else:
        st.warning("Odd Justa não calculada. Registre o primeiro evento para iniciar a análise.")
        
else:
    st.info("Aguardando a inserção das odds iniciais para começar.")
