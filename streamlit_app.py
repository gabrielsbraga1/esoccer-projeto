import streamlit as st
import numpy as np
import pandas as pd

# --- Configurações Iniciais ---
st.set_page_config(layout="wide", page_title="Calculadora de Odd Value para Futebol Virtual HT")

# --- Constantes do Modelo Heurístico (Ajuste a seu critério) ---
# Fatores de impacto nos Gols Esperados (Expected Goals - EG) para um único evento
PESOS = {
    # Revertido para 0.15, conforme solicitado.
    'CHUTE_GOL': 0.15,          
    # Peso muito baixo para Ataque Perigoso (volume).
    'ATAQUE_PERIGOSO': 0.0125,    
    # Peso ligeiramente maior que Ataque Perigoso.
    'ESCANTEIO': 0.02           
}
MAX_MINUTO_HT = 6 # Limite máximo para o primeiro tempo virtual

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
        st.session_state.odd_live_aposta = 2.0 
        st.session_state.linha_gol_selecionada = 2.5 
        # Novas variáveis para Odd Live do confronto
        st.session_state.odd_live_casa = 0.0
        st.session_state.odd_live_empate = 0.0
        st.session_state.odd_live_fora = 0.0


def calcular_probabilidade_inicial(odd):
    """Converte odd para probabilidade implícita."""
    if odd > 1.0:
        return 1 / odd
    return 0.0

def calcular_eg_base(prob_win):
    """Calcula um EG base inicial a partir da probabilidade de vitória (Simplificação)."""
    return 2.0 * prob_win

def calcular_odd_justa_over(eg_total, minuto_atual, linha):
    """
    Calcula a probabilidade e a Odd Justa para o Over X.5 gols, ajustado pelo tempo restante.
    """
    # Usamos o minuto atual como o minuto em que a análise está sendo feita (e o tempo já passou)
    minuto_para_ajuste = minuto_atual - 1 if minuto_atual > 0 else 0

    if minuto_para_ajuste >= 6: # Se passou dos 6 minutos (fim do HT), o fator de tempo é 0
        tempo_restante_fator = 0.0
    elif minuto_para_ajuste == 0:
        tempo_restante_fator = 1.0
    else:
        # Fator de ajuste baseado no tempo total de 90 minutos do jogo real
        tempo_restante_fator = (90 - minuto_para_ajuste) / 90.0
        
    eg_total_ajustado = eg_total * tempo_restante_fator
    
    # Heurística Sigmóide: Transforma EG Total Ajustado em probabilidade Over
    # O '1.5' é um fator de sensibilidade.
    try:
        prob_over = 1 / (1 + np.exp(-(eg_total_ajustado - linha) * 1.5))
        prob_over = np.clip(prob_over, 0.01, 0.99)
    except OverflowError:
        prob_over = 0.99 # Caso o EG seja muito alto

    if prob_over < 0.01:
        return 100.0, 0.0
        
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
    st.session_state.minuto_atual = 1 # O próximo evento será no Minuto 1
    st.session_state.eventos_registrados = []

    # Cálculo das Probabilidades (Normalização)
    p_casa = calcular_probabilidade_inicial(odd_casa)
    p_empate = calcular_probabilidade_inicial(odd_empate)
    p_fora = calcular_probabilidade_inicial(odd_fora)
    soma_p = p_casa + p_empate + p_fora
    p_casa_norm = p_casa / soma_p
    p_fora_norm = p_fora / soma_p

    # Atribui EG Base inicial 
    eg_casa_base = calcular_eg_base(p_casa_norm)
    eg_fora_base = calcular_eg_base(p_fora_norm)
    
    st.session_state.prob_casa_inicial = p_casa_norm
    st.session_state.prob_fora_inicial = p_fora_norm
    
    # Define o EG total inicial como a soma dos EGs base
    st.session_state.eg_casa = eg_casa_base
    st.session_state.eg_fora = eg_fora_base
    st.session_state.eg_total = eg_casa_base + eg_fora_base
    
    # Inicializa as odds live do confronto com as odds pré-jogo
    st.session_state.odd_live_casa = odd_casa
    st.session_state.odd_live_empate = odd_empate
    st.session_state.odd_live_fora = odd_fora
    
    st.session_state.jogo_iniciado = True

def registrar_evento(minuto, chutes_casa, chutes_fora, ataques, escanteios, gols_casa, gols_fora, odd_live_casa, odd_live_empate, odd_live_fora):
    """Processa eventos de um minuto e atualiza o estado do jogo."""
    
    if minuto != st.session_state.minuto_atual:
        st.error(f"O minuto a ser registrado deve ser o Minuto {st.session_state.minuto_atual}.")
        return

    if minuto > MAX_MINUTO_HT:
        st.warning(f"O primeiro tempo virtual já terminou (Max: Minuto {MAX_MINUTO_HT}).")
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
    
    # 4. Atualiza Odds Live do Confronto
    st.session_state.odd_live_casa = odd_live_casa
    st.session_state.odd_live_empate = odd_live_empate
    st.session_state.odd_live_fora = odd_live_fora

    # 5. Registra no Histórico
    odd_minuto, prob_minuto = calcular_odd_justa_over(st.session_state.eg_total, minuto, st.session_state.linha_gol_selecionada)
    
    st.session_state.eventos_registrados.append({
        'Minuto': minuto,
        'Placar': f"{st.session_state.gols_casa} x {st.session_state.gols_fora}",
        'Cht C/F': f"{chutes_casa}/{chutes_fora}",
        'Atq P': ataques,
        'Esc': escanteios,
        'EG Total': f"{st.session_state.eg_total:.2f}",
        f'Odd Justa O{st.session_state.linha_gol_selecionada}': f"{odd_minuto:.2f}"
    })
    
    st.toast(f"Evento do Minuto {minuto} registrado e Odds recalculadas!", icon="⚽")


# --- Inicialização e Layout ---

inicializar_estado()

linha_options = [0.5, 1.5, 2.5]

st.title("⚽ Odd Value Dinâmico HT (Futebol Virtual)")
st.caption(f"Análise e Recálculo em tempo real (Minuto 1 ao {MAX_MINUTO_HT}).")

# --- Seção 1: Configuração Inicial (Sidebar) ---
with st.sidebar:
    st.header("Configuração Inicial (Pré-jogo)")
    
    if st.session_state.jogo_iniciado:
        st.success(f"Jogo Iniciado! Próximo registro: Minuto {st.session_state.minuto_atual}")
    
    col1_s, col2_s, col3_s = st.columns(3)
    odd_casa = col1_s.number_input("Odd Inicial (Casa)", min_value=1.01, value=2.20, step=0.01, key="odd_c")
    odd_empate = col2_s.number_input("Odd Inicial (Empate)", min_value=1.01, value=3.20, step=0.01, key="odd_e")
    odd_fora = col3_s.number_input("Odd Inicial (Fora)", min_value=1.01, value=3.20, step=0.01, key="odd_f")
    
    if st.button("▶️ Iniciar Novo Jogo", key="btn_iniciar"):
        iniciar_jogo(odd_casa, odd_empate, odd_fora)

# --- Seção 2: Jogo em Andamento ---

if st.session_state.jogo_iniciado:
    
    minuto_analisado = st.session_state.minuto_atual - 1
    
    # --- Métricas de Resumo: Placar, EG Total, Linha de Gols ---
    col_score, col_eg, col_line = st.columns([1, 1, 1])
    
    linha_selecionada = col_line.selectbox(
        "Linha de Gols para Análise",
        options=linha_options,
        index=linha_options.index(st.session_state.linha_gol_selecionada) if st.session_state.linha_gol_selecionada in linha_options else 1,
        key="linha_gol_select"
    )
    st.session_state.linha_gol_selecionada = linha_selecionada
    
    col_score.metric(
        label="Placar Atual (C x F)",
        value=f"{st.session_state.gols_casa} x {st.session_state.gols_fora}"
    )
    
    col_eg.metric(
        label="EG Total Acumulado",
        value=f"{st.session_state.eg_total:.2f}"
    )

    st.markdown("---")
    st.subheader(f"Odds e Forças Atuais (Análise no Minuto {minuto_analisado})")
    
    # Odds Live do Confronto (Casa/Empate/Fora)
    col_oc, col_oe, col_of = st.columns(3)
    col_oc.metric("Odd Casa", f"{st.session_state.odd_live_casa:.2f}")
    col_oe.metric("Odd Empate", f"{st.session_state.odd_live_empate:.2f}")
    col_of.metric("Odd Fora", f"{st.session_state.odd_live_fora:.2f}")

    st.markdown("---")
    st.subheader(f"Registro de Eventos no Minuto {st.session_state.minuto_atual}")
    
    if st.session_state.minuto_atual > MAX_MINUTO_HT:
        st.warning(f"O primeiro tempo virtual terminou no Minuto {MAX_MINUTO_HT}. Reinicie para um novo jogo.")
    else:
        with st.form("registro_eventos"):
            
            # Minuto e Odd Live Over/Under
            col_m, col_odd_live = st.columns(2)
            
            minuto_registro = col_m.number_input(
                "Minuto a ser Registrado", 
                min_value=st.session_state.minuto_atual, 
                max_value=MAX_MINUTO_HT, 
                value=st.session_state.minuto_atual,
                disabled=True # Força o avanço sequencial
            )
            
            live_odd_aposta = col_odd_live.number_input(
                f"Odd Live da Casa (Over {linha_selecionada})", 
                min_value=1.01, 
                value=st.session_state.odd_live_aposta, 
                step=0.01,
                key="live_odd_input_form"
            )
            # Armazena o último input de odd Over/Under para o próximo ciclo
            st.session_state.odd_live_aposta = live_odd_aposta
            
            st.markdown("#### Odds do Confronto (Live)")
            col_oc, col_oe, col_of = st.columns(3)
            
            odd_lc = col_oc.number_input("Odd Live (Casa)", min_value=1.01, value=st.session_state.odd_live_casa, step=0.01, key="odd_live_c")
            odd_le = col_oe.number_input("Odd Live (Empate)", min_value=1.01, value=st.session_state.odd_live_empate, step=0.01, key="odd_live_e")
            odd_lf = col_of.number_input("Odd Live (Fora)", min_value=1.01, value=st.session_state.odd_live_fora, step=0.01, key="odd_live_f")

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
            
            if st.form_submit_button("🔁 Recalcular Tendência e Registrar"):
                # Passa as novas odds do confronto para serem armazenadas
                registrar_evento(minuto_registro, chutes_casa, chutes_fora, ataques_perigosos, escanteios, gols_casa, gols_fora, odd_lc, odd_le, odd_lf)

    st.markdown("---")
    st.subheader(f"📊 Análise de Odd Value para Over {linha_selecionada} Gols")
    
    # Recalcula a Odd Justa e Probabilidade no estado ATUAL (antes do registro do próximo minuto)
    odd_justa_atual, prob_justa_atual = calcular_odd_justa_over(st.session_state.eg_total, minuto_analisado, linha_selecionada)
    
    if odd_justa_atual > 0:
        
        # Cálculo do Valor (Value)
        value_ratio = st.session_state.odd_live_aposta / odd_justa_atual
        
        col_prob, col_justa, col_value = st.columns(3)
        
        col_prob.metric(
            label=f"Prob. Justa Over {linha_selecionada} (%)",
            value=f"{prob_justa_atual * 100:.1f}%"
        )
        
        col_justa.metric(
            label=f"Odd Justa/Esperada",
            value=f"{odd_justa_atual:.2f}",
            delta=f"Live Odd: {st.session_state.odd_live_aposta:.2f}",
            delta_color="off" 
        )
        
        # Interpretação do Odd Value
        if value_ratio > 1.05:
            value_status = "✅ Forte Valor! (Value)"
            color = "#10B981" # Green
        elif value_ratio > 1.01:
            value_status = "🟡 Pequeno Valor"
            color = "#F59E0B" # Yellow/Orange
        else:
            value_status = "❌ Sem Valor (No Value)"
            color = "#EF4444" # Red
            
        col_value.markdown(
            f"""
            <div style='background-color: {color}; padding: 15px; border-radius: 8px; color: white; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;'>
                <h4 style='margin-top: 0; font-size: 1.2em;'>Odd Value Ratio</h4>
                <p style='font-size: 2em; margin: 0;'>**{value_ratio:.2f}**</p>
                <p style='margin: 0; font-weight: bold;'>{value_status}</p>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("Histórico de Eventos Registrados")
        if st.session_state.eventos_registrados:
            df_historico = pd.DataFrame(st.session_state.eventos_registrados)
            st.dataframe(df_historico, use_container_width=True)
        
    else:
        st.info("Inicie o jogo e registre o primeiro minuto para ver a análise de Odd Value.")
        
else:
    st.info("Aguardando a inserção das odds iniciais para começar.")
