"""
Simplex Tableau — interface Streamlit para Programação Linear.
Suporta maximização/minimização, 2 a 4 variáveis (interface), restrições <=, >=, =,
Simplex em duas fases, tableaus, preços-sombra e intervalos de viabilidade do RHS.
Instalação: pip install streamlit numpy pandas
Execução: streamlit run simplex_tableau.py
"""
import numpy as np
import pandas as pd


TOL = 1e-9
MAX_ITER = 1000


def pivotar(tab, base, linha, coluna, H):
    H[linha] /= tab[linha, coluna]
    tab[linha, :] /= tab[linha, coluna]
    for i in range(tab.shape[0]):
        if i != linha:
            if i < len(base): H[i] -= tab[i, coluna] * H[linha]
            tab[i, :] -= tab[i, coluna] * tab[linha, :]
    base[linha] = coluna


def otimizar(tab, base, custos, nomes, fase, historico, H):
    m = len(base)
    tab[m, :] = 0.0
    tab[m, :-1] = -custos
    for i, j in enumerate(base):
        tab[m, :] += custos[j] * tab[i, :]
    for it in range(MAX_ITER):
        candidatos = [j for j in range(tab.shape[1] - 1) if tab[m, j] < -TOL]
        if not candidatos:
            return "otimo"
        entra = min(candidatos)  # Bland
        possiveis = [i for i in range(m) if tab[i, entra] > TOL]
        if not possiveis:
            return "ilimitado"
        razoes = [(tab[i, -1] / tab[i, entra], base[i], i) for i in possiveis]
        menor = min(x[0] for x in razoes)
        sai = min((x for x in razoes if abs(x[0] - menor) <= TOL), key=lambda x: x[1])[2]
        pivotar(tab, base, sai, entra, H)
        historico.append((f"Fase {fase} · iteração {it+1}", tab.copy(), base.copy(), nomes.copy()))
    return "limite_iteracoes"


def simplex_tableau(c, A, b, sinais, objetivo="Maximizar"):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()
    m, n = A.shape
    sinais = list(sinais)
    if len(c) != n or len(b) != m or len(sinais) != m:
        raise ValueError("Confira a quantidade de coeficientes e restrições.")
    if not (np.isfinite(c).all() and np.isfinite(A).all() and np.isfinite(b).all()):
        raise ValueError("Preencha todos os campos com números finitos.")
    original_b = b.copy()
    H = np.eye(m)
    removed = []
    for i in range(m):
        if b[i] < -TOL:
            H[i] *= -1
            A[i, :] *= -1
            b[i] *= -1
            sinais[i] = {"≤": "≥", "≥": "≤", "=": "="}[sinais[i]]

    colunas = [A]
    nomes = [f"x{j+1}" for j in range(n)]
    base = [-1] * m
    artificiais = []
    slacks = {}
    for i, sinal in enumerate(sinais):
        if sinal == "≤":
            col = np.zeros((m, 1)); col[i, 0] = 1
            colunas.append(col); nomes.append(f"f{i+1}")
            base[i] = len(nomes)-1; slacks[i] = nomes[-1]
        elif sinal == "≥":
            exc = np.zeros((m, 1)); exc[i, 0] = -1
            colunas.append(exc); nomes.append(f"e{i+1}")
            art = np.zeros((m, 1)); art[i, 0] = 1
            colunas.append(art); nomes.append(f"a{i+1}")
            base[i] = len(nomes)-1; artificiais.append(base[i])
        elif sinal == "=":
            art = np.zeros((m, 1)); art[i, 0] = 1
            colunas.append(art); nomes.append(f"a{i+1}")
            base[i] = len(nomes)-1; artificiais.append(base[i])
        else:
            raise ValueError(f"Sinal inválido na restrição {i+1}.")
    M = np.hstack(colunas)
    tab = np.zeros((m+1, M.shape[1]+1))
    tab[:m, :-1] = M
    tab[:m, -1] = b
    hist = [("Tableau inicial", tab.copy(), base.copy(), nomes.copy())]

    if artificiais:
        c1 = np.zeros(M.shape[1]); c1[artificiais] = -1
        status = otimizar(tab, base, c1, nomes, 1, hist, H)
        if status != "otimo":
            return {"status": status, "historico": hist}
        if tab[m, -1] < -1e-7:
            return {"status": "inviavel", "historico": hist}
        i = 0
        while i < len(base):
            if base[i] in artificiais:
                cand = [j for j in range(M.shape[1]) if j not in artificiais and abs(tab[i,j]) > TOL]
                if cand:
                    pivotar(tab, base, i, cand[0], H); i += 1
                else:
                    removed.append(H[i].copy())
                    H = np.delete(H, i, axis=0)
                    tab = np.delete(tab, i, axis=0); base.pop(i)
            else:
                i += 1
        manter = [j for j in range(M.shape[1]) if j not in artificiais]
        tab = tab[:, manter + [M.shape[1]]]
        mapa = {old:new for new,old in enumerate(manter)}
        base = [mapa[j] for j in base]
        M = M[:, manter]
        nomes = [nomes[j] for j in manter]

    c2 = c.copy() if objetivo == "Maximizar" else -c.copy()
    custos = np.zeros(M.shape[1]); custos[:n] = c2
    status = otimizar(tab, base, custos, nomes, 2, hist, H)
    if status != "otimo":
        return {"status": status, "historico": hist}
    x = np.zeros(n)
    for i,j in enumerate(base):
        if j < n: x[j] = tab[i,-1]
    z = float(np.dot(c,x))
    hist.append(("Tableau ótimo", tab.copy(), base.copy(), nomes.copy()))
    D = H
    E = np.asarray(removed).reshape(-1,m)
    factor = 1 if objetivo == "Maximizar" else -1
    prices = factor * (custos[base] @ D)
    reduced = -factor * tab[-1,:-1][:n]
    cost_ranges = []
    nonbasic = [j for j in range(len(nomes)) if j not in base]
    for j in range(n):
        slope = np.zeros(len(nomes))
        if j in base: slope = factor * tab[base.index(j), :-1].copy()
        slope[j] -= factor
        lo, hi = interval(tab[-1, nonbasic], slope[nonbasic])
        cost_ranges.append((c[j]+lo,c[j]+hi))
    return {"status":"otimo", "historico":hist, "base":base, "nomes":nomes,
            "D":D, "E":E, "prices":prices, "reduced":reduced,
            "cost_ranges":cost_ranges, "n":n, "b":original_b}


def interval(values, slopes):
    lo, hi = -np.inf, np.inf
    for v, a in zip(values, slopes):
        if a > TOL: lo = max(lo, -max(0,v)/a)
        elif a < -TOL: hi = min(hi, -max(0,v)/a)
    return lo, hi


def evaluate(r, b):
    b = np.asarray(b, float)
    v = r['D'] @ b
    valid = np.all(v >= -1e-7) and np.all(np.abs(r['E'] @ b) <= 1e-7)
    x = np.zeros(r['n'])
    for i,j in enumerate(r['base']):
        if j < len(x): x[j] = v[i]
    ranges = []
    for i in range(len(b)):
        lo, hi = interval(v, r['D'][:,i])
        if np.any(np.abs(r['E'][:,i]) > TOL): lo = hi = 0
        ranges.append((b[i]+lo,b[i]+hi))
    return valid, x, float(r['prices'] @ b), ranges


def app():
    import streamlit as st
    st.set_page_config(page_title="Simplex Studio", page_icon="📊", layout="wide")
    st.markdown("""<style>
    .block-container {max-width:1200px;padding-top:2rem;}
    [data-testid=stSidebar] {background:#e8eef8;}
    [data-testid=stSidebar] * {color:#17243b !important;}
    [data-testid=stSidebar] input {background:white!important;color:#17243b!important;}
    .hero {background:linear-gradient(120deg,#14233d,#285c99);color:white;padding:28px;border-radius:20px;margin-bottom:24px;}
    .hero h1 {color:white;margin:0;}.hero p {color:#dceaff;margin:8px 0 0;}
    [data-testid=stMetric] {border:1px solid #b8c8df;border-radius:14px;padding:16px;}
    </style><div class="hero"><h1>Simplex Studio</h1><p>Pesquisa operacional • Otimização e análise de sensibilidade</p></div>""",unsafe_allow_html=True)
    with st.sidebar:
        st.header("Configuração")
        n=st.selectbox("Número de variáveis",[2,3,4],key="n")
        m=st.number_input("Número de restrições",min_value=1,value=3,step=1,key="m")
        direction=st.radio("Objetivo",["Maximizar","Minimizar"])
        st.caption("Todas as variáveis são não negativas.")
    names=[f"x{i+1}" for i in range(n)]
    st.subheader("01 · Função objetivo")
    cdf=st.data_editor(pd.DataFrame([[3.,5.]+[0.]*(n-2)],columns=names),key=f"c{n}",hide_index=True,width="stretch")
    st.subheader("02 · Restrições e variações iniciais")
    example=[[1.,0.,4.],[0.,2.,12.],[3.,2.,18.]]
    rows=[]
    for i in range(m):
        a=example[i][:2]+[0.]*(n-2) if i<3 else [0.]*n
        rows.append(a+["≤",example[i][2] if i<3 else 0.,0.])
    df=st.data_editor(pd.DataFrame(rows,columns=names+["Sinal","LD","Variação Δ"],index=[f"R{i+1}" for i in range(m)]),key=f"a{n}_{m}",width="stretch",column_config={"Sinal":st.column_config.SelectboxColumn(options=["≤","≥","="],required=True)})
    st.caption("LD = lado direito. Variação Δ = quanto deseja acrescentar ou retirar do LD.")
    c=cdf.iloc[0].to_numpy(float); A=df[names].to_numpy(float); b=df.LD.to_numpy(float); signs=df.Sinal.tolist()
    signature=repr((c.tolist(),A.tolist(),b.tolist(),signs,direction))
    if st.button("Resolver pelo Simplex Tableau",type="primary",width="stretch"):
        st.session_state.pop("result",None)
        try:
            r=simplex_tableau(c,A,b,signs,direction)
            st.session_state.result=(signature,r)
            st.session_state.current=b.copy()
            st.session_state.log=[]
            st.session_state.version=st.session_state.get("version",0)+1
            if r['status']=='otimo':
                candidate=b+df['Variação Δ'].to_numpy(float)
                if np.any(candidate!=b): apply_change(st,r,candidate)
        except Exception as e: st.error(str(e))
    if 'result' not in st.session_state: return
    sig,r=st.session_state.result
    if sig!=signature:
        st.info("Modelo alterado. Clique em Resolver para iniciar uma nova análise."); return
    if r['status']!='otimo':
        st.error({'inviavel':'Problema inviável','ilimitado':'Problema ilimitado'}.get(r['status'],r['status'])); return
    valid,x,z,ranges=evaluate(r,st.session_state.current)
    st.subheader("03 · Solução ótima")
    _,x0,z0,_=evaluate(r,b)
    cols=st.columns(3); cols[0].metric("Z original",f"{z0:.6g}"); cols[1].metric("Z atual",f"{z:.6g}"); cols[2].metric("Status","Ótimo")
    st.dataframe(pd.DataFrame({'Variável':names,'Valor original':x0,'Valor atual':x}),hide_index=True,width="stretch")
    st.subheader("04 · Sensibilidade das restrições")
    current=st.session_state.current
    slack=np.array([current[i]-v if signs[i]=='≤' else v-current[i] if signs[i]=='≥' else 0 for i,v in enumerate(A@x)])
    st.dataframe(pd.DataFrame({'Restrição':[f'R{i+1}' for i in range(m)],'Preço-sombra':r['prices'],'Folga / excesso':slack,'LD original':b,'LD atual':current,'Limite inferior':[v[0] for v in ranges],'Limite superior':[v[1] for v in ranges]}).round(6),hide_index=True,width="stretch")
    st.caption("Faixas individuais: altere um LD por vez. Para alterações simultâneas, o programa verifica a base conjuntamente. Fora da faixa, o preço-sombra não garante o ótimo.")
    st.subheader("05 · Sensibilidade da função objetivo")
    st.dataframe(pd.DataFrame({'Variável':names,'Valor ótimo':x,'Custo reduzido':r['reduced'],'Coeficiente':c,'Limite inferior':[v[0] for v in r['cost_ranges']],'Limite superior':[v[1] for v in r['cost_ranges']]}).round(6),hide_index=True,width="stretch")
    st.subheader("06 · Novas alterações no lado direito")
    st.caption("Informe os novos valores absolutos. Nenhuma nova execução do Simplex é feita nesta etapa.")
    with st.form('changes'):
        changes=st.data_editor(pd.DataFrame({'Novo LD':current},index=[f'R{i+1}' for i in range(m)]),key=f"changes{st.session_state.version}",width="stretch")
        submitted=st.form_submit_button("Analisar alterações",width="stretch")
    if submitted:
        apply_change(st,r,changes['Novo LD'].to_numpy(float)); st.rerun()
    if st.session_state.log:
        last=st.session_state.log[-1]
        (st.success if last['Válida'] else st.warning)(last['Resultado'])
        st.dataframe(pd.DataFrame(st.session_state.log),hide_index=True,width="stretch")
    with st.expander("Tableaus e iterações da resolução original"):
        for title,tab,base,labels in r['historico']:
            st.write(title)
            st.dataframe(pd.DataFrame(tab,columns=labels+['LD'],index=[labels[j] for j in base]+['Z']).round(6),width="stretch")


def apply_change(st,r,candidate):
    old=st.session_state.current
    valid,x,z,ranges=evaluate(r,candidate)
    valid=bool(valid and np.isfinite(candidate).all())
    delta=float(r['prices']@(candidate-old))
    if valid:
        message=f"Alteração válida na base ótima. ΔZ = {delta:.6g}; novo Z = {z:.6g}."
        st.session_state.current=candidate.copy()
    else:
        message="Fora da faixa de validade da base ótima. Não é possível garantir o novo ótimo pelo preço-sombra. Valores anteriores mantidos."
    st.session_state.log.append({'Válida':valid,'LD solicitado':str(candidate.tolist()),'Novo Z':z if valid else None,'Resultado':message})
    st.session_state.version+=1


if __name__=='__main__': app()
