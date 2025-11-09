import streamlit as st
import numpy as np

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
        st.session_state.minuto_atual = 0
        st.session_state.eventos_registrados = []
        st.session_state.eg_casa = 0.0
        st.session_state.eg_fora = 0.0
        st.session_state.eg_total = 0.0

def calcular_probabilidade_inicial(odd):
    """Converte odd para probabilidade implícita."""
    if odd > 1.0:
        return 1 / odd
    return 0.0

def calcular_eg_base(prob_win):
    """Calcula um EG base inicial a partir da probabilidade de vitória (Simplificação)."""
    # Se a probabilidade de vitória for 0.5 (odd 2.0), EG_Base será 1.0.
    return 2.0 * prob_win

def calcular_odd_justa_over(eg_total, linha=LINHA_GOL):
    """
    Calcula a probabilidade do Over X.5 gols usando uma simplificação do modelo de Poisson.
    Aqui, usamos a distribuição normal para uma aproximação rápida.
    Em um modelo real, seria necessário calcular a soma das probabilidades de P(k=gols) para k > linha.
    """
    # Simplificação: Baseamos a probabilidade Over/Under no EG Total, ajustado pelo tempo.
    # Assumimos que a proporção dos EG iniciais é mantida no tempo.
    
    # Exemplo Heurístico:
    # Se EG Total = 3.0, a chance de Over 2.5 é alta.
    
    # 1. Ajuste do EG pelo tempo restante (quanto mais tarde, menos tempo para marcar)
    minuto = st.session_state.minuto_atual
    tempo_restante_fator = (90 - minuto) / 90.0
    eg_total_ajustado = eg_total * tempo_restante_fator
    
    # 2. Heurística para Over 2.5 (probabilidade aumenta com o EG Total)
    # np.clip garante que a probabilidade esteja entre 0.05 e 0.95
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
    """Calcula probabilidades iniciais e EG base."""
    
    # Validação de Odds
    if odd_casa <= 1.0 or odd_empate <= 1.0 or odd_fora <= 1.0:
        st.error("Odds devem ser maiores que 1.0.")
        return

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
    st.session_state.minuto_atual = 1

def registrar_evento(minuto, chutes, ataques, escanteios, gols_casa, gols_fora):
    """Processa eventos de um minuto e atualiza o estado do jogo."""
    
    if minuto <= st.session_state.minuto_atual:
        st.error(f"O minuto deve ser superior ao minuto atual ({st.session_state.minuto_atual}).")
        return

    # 1. Cálculo do EG adicional (Heurística)
    eg_adicional_casa = 0
    eg_adicional_fora = 0
    
    # Assume que a equipe que atacou é a equipe Casa se ambas as entradas (chutes/ataques) não forem nulas
    # Se Chutes a Gol > 0, o impacto é no EG da equipe que chuta (Casa ou Fora)
    
    # Para simplificar, vou assumir que a equipe que marcou o gol (se houver) é a que atacou,
    # e o impacto dos ataques/chutes é distribuído proporcionalmente ao EG inicial.
    
    # Para ser mais preciso, precisaria saber quem fez o chute/ataque.
    # Vamos assumir que os chutes/ataques registrados são *para* a equipe Casa OU Fora (o usuário precisa especificar)
    
    # --- Adaptação do Input para o Modelo ---
    # Para o modelo funcionar, o input do evento precisa ser atribuído a um time.
    # Simplificando a interface, vou assumir que o impacto é genérico e distribuído pela força inicial.
    
    # Impacto Genérico (Distribuído pela Força Relativa)
    if st.session_state.eg_total > 0:
        fator_casa = st.session_state.eg_casa / st.session_state.eg_total
        fator_fora = st.session_state.eg_fora / st.session_state.eg_total
    else:
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
    st.session_state.eg_casa += eg_adicional_casa
    st.session_state.eg_fora += eg_adicional_fora
    st.session_state.eg_total = st.session_state.eg_casa + st.session_state.eg_fora
    
    # 4. Registra e exibe evento
    st.session_state.eventos_registrados.append({
        'Minuto': minuto,
        'Gols C/F': f"{gols_casa}/{gols_fora}",
        'Chutes': chutes,
        'Ataques': ataques,
        'Escanteios': escanteios,
        'EG Casa': f"+{eg_adicional_casa:.2f}",
        'EG Fora': f"+{eg_adicional_fora:.2f}"
    })
    
    st.toast("Evento registrado e Odds recalculadas!", icon="⚽")
    
# --- Inicialização e Layout ---

inicializar_estado()

st.title("⚽ Odd Value Dinâmico (Futebol Virtual)")
st.caption("Modelo Heurístico Simplificado para Recalcular a Probabilidade de Gols.")

# --- Seção 1: Configuração Inicial ---
with st.sidebar:
    st.header("Configuração Inicial (Pré-jogo)")
    
    if st.session_state.jogo_iniciado:
        st.warning("Jogo em andamento. Para um novo jogo, recarregue a página.")
    else:
        col1_s, col2_s, col3_s = st.columns(3)
        odd_casa = col1_s.number_input("Odd Inicial (Casa)", min_value=1.01, value=2.20, step=0.01)
        odd_empate = col2_s.number_input("Odd Inicial (Empate)", min_value=1.01, value=3.20, step=0.01)
        odd_fora = col3_s.number_input("Odd Inicial (Fora)", min_value=1.01, value=3.20, step=0.01)
        
        if st.button("▶️ Iniciar Jogo e Calcular Força Base"):
            iniciar_jogo(odd_casa, odd_empate, odd_fora)

# --- Seção 2: Jogo em Andamento ---

if st.session_state.jogo_iniciado:
    
    st.header(f"Minuto Atual: {st.session_state.minuto_atual}")
    
    # Display Score and Initial Force
    col_score, col_eg, col_odds = st.columns(3)
    
    col_score.metric(
        label="Placar (Casa x Fora)",
        value=f"{st.session_state.gols_casa} x {st.session_state.gols_fora}"
    )
    
    col_eg.metric(
        label="EG Total Acumulado (Expected Goals)",
        value=f"{st.session_state.eg_total:.2f} Gols"
    )

    prob_casa_perc = st.session_state.prob_casa_inicial * 100
    prob_fora_perc = st.session_state.prob_fora_inicial * 100
    
    col_odds.markdown(
        f"""
        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
            <h4 style='margin-top: 0; font-size: 1em;'>Força Base (Normalizada)</h4>
            <p style='margin: 0;'>Casa: **{prob_casa_perc:.1f}%** | Fora: **{prob_fora_perc:.1f}%**</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.subheader("Registro de Eventos em Campo")
    
    with st.form("registro_eventos"):
        
        col_m, col_chute, col_atq, col_esc = st.columns([1, 1.5, 1.5, 1.5])
        
        minuto_novo = col_m.number_input(
            "Minuto do Evento", 
            min_value=st.session_state.minuto_atual + 1, 
            max_value=90, 
            value=st.session_state.minuto_atual + 1
        )
        
        # Chutes a gol (Obrigatório, mas setado como 0 para permitir outros eventos)
        chutes_gol = col_chute.number_input("Chutes a Gol (Obrigatório)", min_value=0, value=0, step=1)
        ataques_perigosos = col_atq.number_input("Ataques Perigosos (Opcional)", min_value=0, value=0, step=1)
        escanteios = col_esc.number_input("Escanteios (Opcional)", min_value=0, value=0, step=1)
        
        st.markdown("---")
        st.markdown("**Gols Marcados**")
        col_gol_casa, col_gol_fora = st.columns(2)
        
        gols_casa = col_gol_casa.number_input("Gols do Time Casa (+)", min_value=0, value=0, step=1)
        gols_fora = col_gol_fora.number_input("Gols do Time Fora (+)", min_value=0, value=0, step=1)
        
        if st.form_submit_button("🔁 Recalcular Tendência"):
            if chutes_gol == 0 and gols_casa == 0 and gols_fora == 0:
                 st.error("Pelo menos um 'Chute a Gol' ou um 'Gol' é obrigatório para recalcular.")
            else:
                registrar_evento(minuto_novo, chutes_gol, ataques_perigosos, escanteios, gols_casa, gols_fora)

    st.subheader(f"📊 Análise de Odd Value (Over {LINHA_GOL})")
    
    odd_justa, prob_justa = calcular_odd_justa_over(st.session_state.eg_total)
    
    if odd_justa > 0:
        
        # Input da Odd da Casa de Apostas
        live_odd_aposta = st.number_input(
            f"Odd Atual da Casa de Apostas (Over {LINHA_GOL} Gols)", 
            min_value=1.01, 
            value=2.0, 
            step=0.01,
            key="live_odd_input"
        )
        
        # Cálculo do Valor (Value)
        value_ratio = live_odd_aposta / odd_justa
        
        st.markdown("---")
        
        col_odd, col_prob, col_value = st.columns(3)
        
        # Exibição dos Resultados
        col_odd.metric(
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
        st.subheader("Histórico de Eventos")
        if st.session_state.eventos_registrados:
            # Exibir a tabela de histórico de eventos
            st.dataframe(st.session_state.eventos_registrados, use_container_width=True)
        else:
            st.info("Nenhum evento registrado ainda.")
            
    else:
        st.warning("Aguardando o registro do primeiro evento para calcular o Odd Justo.")
        
else:
    st.info("Aguardando a inserção das odds iniciais para começar.")
