"""
Projeto de Análise de Dados em Python (Av3 - entrega final)
Dataset: Movies (vega-datasets), 3.201 filmes com orçamento, bilheteria,
gênero e notas da crítica (Rotten Tomatoes) e do público (IMDB).

Pergunta central: o orçamento, o gênero e as notas da crítica e do público
estão relacionados ao sucesso de bilheteria dos filmes?

Hipóteses:
  H1: quanto maior o orçamento, maior a bilheteria mundial.
  H2: a crítica e o público costumam concordar nas notas.
  H3: a nota do público tem pouca relação com a bilheteria.
  H4: o retorno sobre o orçamento muda com o gênero (terror rende mais que drama).

Como executar:
  pip install pandas numpy scipy matplotlib seaborn
  python analise_filmes_final.py
O script baixa o movies.json se ele não estiver na pasta, gera os gráficos
na pasta "graficos/" e salva os resultados em "resultados.txt".
"""

import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats

# ----------------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------------
CAMINHO_JSON = "movies.json"
URL_DADOS = "https://raw.githubusercontent.com/vega/vega-datasets/main/data/movies.json"
PASTA_GRAFICOS = "graficos"
SEMENTE = 42                      # garante o mesmo resultado no bootstrap
AZUL, VERMELHO = "#1F6F8B", "#C8553D"

TRADUCAO_GENEROS = {
    "Documentary": "Documentário", "Horror": "Terror", "Musical": "Musical",
    "Adventure": "Aventura", "Romantic Comedy": "Comédia romântica",
    "Action": "Ação", "Comedy": "Comédia", "Thriller/Suspense": "Suspense",
    "Black Comedy": "Humor negro", "Drama": "Drama", "Western": "Faroeste",
}

LINHAS_RESULTADO = []             # guarda tudo o que é impresso, para salvar em arquivo


def imprimir(texto: str = "") -> None:
    """Imprime na tela e guarda o texto para o arquivo resultados.txt."""
    print(texto)
    LINHAS_RESULTADO.append(str(texto))


def formatar_p(p: float) -> str:
    """Formata o valor-p: valores muito pequenos aparecem como '< 0,001'."""
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def rotulo_dolar(valor, _posicao=None) -> str:
    """Rótulo de eixo em dólares: 1 mi, 100 mi, 1 bi."""
    if valor >= 1e9:
        return f"US$ {valor / 1e9:g} bi"
    if valor >= 1e6:
        return f"US$ {valor / 1e6:g} mi"
    if valor >= 1e3:
        return f"US$ {valor / 1e3:g} mil"
    return f"US$ {valor:g}"


# ----------------------------------------------------------------------------
# 1. Leitura e inspeção
# ----------------------------------------------------------------------------
def baixar_dados_se_necessario(caminho: str, url: str) -> None:
    """Baixa o arquivo de dados se ele ainda não estiver na pasta."""
    if not os.path.exists(caminho):
        print(f"Baixando {caminho}...")
        urllib.request.urlretrieve(url, caminho)


def carregar_dados(caminho: str) -> pd.DataFrame:
    """Lê o arquivo JSON e devolve um DataFrame."""
    return pd.read_json(caminho)


def inspecionar_dados(df: pd.DataFrame) -> None:
    """Primeira inspeção: tamanho, tipos, ausentes e duplicados."""
    imprimir("=== INSPEÇÃO INICIAL ===")
    imprimir(f"Dimensões (linhas, colunas): {df.shape}")
    imprimir("\nValores ausentes por coluna (só as que têm ausentes):")
    ausentes = df.isna().sum()
    imprimir(ausentes[ausentes > 0].sort_values(ascending=False).to_string())
    imprimir(f"\nLinhas duplicadas: {df.duplicated().sum()}")


# ----------------------------------------------------------------------------
# 2. Limpeza e tratamento
# ----------------------------------------------------------------------------
def tratar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e prepara os dados. Cada decisão está comentada."""
    df = df.copy()

    # 1) Duplicados: remove linhas repetidas (o dataset não tem, mas garantimos).
    df = df.drop_duplicates()

    # 2) Título ausente: sem título não dá para identificar o filme, então removemos.
    df = df.dropna(subset=["Title"])

    # 3) Datas: converte texto para data. Alguns filmes antigos aparecem com ano
    #    no futuro (ex.: 2039 em vez de 1939), então subtraímos 100 anos desses casos.
    df["Release Date"] = pd.to_datetime(df["Release Date"], format="%b %d %Y")
    ano = df["Release Date"].dt.year
    df["Ano"] = ano.where(ano <= 2011, ano - 100)
    df["Decada"] = (df["Ano"] // 10 * 10).astype(int)

    # 4) Bilheteria mundial igual a 0 não é um valor real: tratamos como ausente.
    df["Worldwide Gross"] = df["Worldwide Gross"].replace(0, np.nan)

    # 5) Colunas com muitos vazios (mais de 40%) são removidas.
    df = df.drop(columns=["US DVD Sales", "Running Time min", "Director"])

    # 6) Texto ausente vira "Não informado". Notas ausentes ficam vazias
    #    (inventar nota distorceria as análises).
    for coluna in ["Major Genre", "MPAA Rating", "Distributor", "Source", "Creative Type"]:
        df[coluna] = df[coluna].fillna("Não informado")
    for coluna in ["Major Genre", "MPAA Rating"]:
        df[coluna] = df[coluna].astype("category")

    # 7) Novas variáveis: lucro, retorno (bilheteria / orçamento) e nota do IMDB em escala 0-100.
    df["Lucro"] = df["Worldwide Gross"] - df["Production Budget"]
    df["ROI"] = df["Worldwide Gross"] / df["Production Budget"]
    df["IMDB 0-100"] = df["IMDB Rating"] * 10
    return df


def filmes_com_retorno(df: pd.DataFrame) -> pd.DataFrame:
    """Filmes com orçamento e bilheteria válidos (base das análises financeiras)."""
    return df.dropna(subset=["ROI"])


# ----------------------------------------------------------------------------
# 3. Estatísticas descritivas
# ----------------------------------------------------------------------------
def estatisticas_descritivas(df: pd.DataFrame) -> None:
    """Resumo numérico das principais variáveis."""
    imprimir("\n=== ESTATÍSTICAS DESCRITIVAS ===")
    validos = filmes_com_retorno(df)
    imprimir(f"Filmes após a limpeza: {len(df)}")
    imprimir(f"Período dos lançamentos: {df['Ano'].min()} a {df['Ano'].max()}")
    imprimir(f"Filmes com orçamento e bilheteria válidos: {len(validos)}")
    colunas = ["Production Budget", "Worldwide Gross", "IMDB Rating", "Rotten Tomatoes Rating"]
    imprimir("\nMediana e média:")
    resumo = validos[colunas].agg(["median", "mean"]).round(2)
    imprimir(resumo.to_string())
    imprimir(f"\nFaturaram mais do que custaram: {(validos['Lucro'] > 0).mean():.1%}")
    imprimir(f"ROI mediano: {validos['ROI'].median():.2f}x")
    maior = df.loc[df["Worldwide Gross"].idxmax()]
    imprimir(f"Maior bilheteria: {maior['Title']} (US$ {maior['Worldwide Gross'] / 1e9:.2f} bi)")
    imprimir(f"Filmes de 1990 em diante: {(df['Ano'] >= 1990).mean():.1%}")


# ----------------------------------------------------------------------------
# 4. Teste das hipóteses
# ----------------------------------------------------------------------------
def teste_h1(df: pd.DataFrame) -> dict:
    """H1: correlação de Spearman e regressão em escala log (orçamento x bilheteria)."""
    v = filmes_com_retorno(df)
    rho, p = stats.spearmanr(v["Production Budget"], v["Worldwide Gross"])
    reg = stats.linregress(np.log10(v["Production Budget"]), np.log10(v["Worldwide Gross"]))
    resultado = {"rho": rho, "p": p, "n": len(v), "inclinacao": reg.slope,
                 "intercepto": reg.intercept, "r2": reg.rvalue ** 2}
    imprimir("\n=== H1: orçamento x bilheteria ===")
    imprimir(f"Spearman: rho = {rho:.2f}, p {formatar_p(p)}, n = {len(v)}")
    imprimir(f"Regressão log-log: inclinação = {reg.slope:.2f}, R² = {reg.rvalue ** 2:.2f}")
    return resultado


def teste_h2(df: pd.DataFrame) -> dict:
    """H2: correlação entre a nota da crítica (Rotten Tomatoes) e a do público (IMDB)."""
    par = df[["Rotten Tomatoes Rating", "IMDB 0-100"]].dropna()
    rho, p = stats.spearmanr(par["Rotten Tomatoes Rating"], par["IMDB 0-100"])
    imprimir("\n=== H2: nota da crítica x nota do público ===")
    imprimir(f"Spearman: rho = {rho:.2f}, p {formatar_p(p)}, n = {len(par)}")
    return {"rho": rho, "p": p, "n": len(par)}


def teste_h3(df: pd.DataFrame) -> dict:
    """H3: relação das notas com a bilheteria, com o retorno (ROI) e com o orçamento."""
    v = filmes_com_retorno(df)
    resultado = {}
    imprimir("\n=== H3: notas x bilheteria ===")
    for nome, coluna in [("IMDB", "IMDB Rating"), ("Crítica", "Rotten Tomatoes Rating")]:
        for alvo, rotulo in [("Worldwide Gross", "bilheteria"), ("ROI", "ROI"),
                             ("Production Budget", "orçamento")]:
            par = v[[coluna, alvo]].dropna()
            rho, p = stats.spearmanr(par[coluna], par[alvo])
            resultado[(nome, rotulo)] = (rho, p, len(par))
            imprimir(f"{nome} x {rotulo}: rho = {rho:.2f}, p {formatar_p(p)}, n = {len(par)}")

    imprimir("\nIMDB x bilheteria dentro de cada gênero (gêneros com 60+ filmes):")
    por_genero = {}
    for genero, grupo in v[v["Major Genre"] != "Não informado"].groupby("Major Genre", observed=True):
        par = grupo[["IMDB Rating", "Worldwide Gross"]].dropna()
        if len(par) >= 60:
            rho = stats.spearmanr(par["IMDB Rating"], par["Worldwide Gross"])[0]
            por_genero[genero] = (rho, len(par))
            imprimir(f"  {TRADUCAO_GENEROS.get(genero, genero)}: rho = {rho:.2f} (n = {len(par)})")
    resultado["por_genero"] = por_genero
    return resultado


def teste_h4(df: pd.DataFrame) -> dict:
    """H4: o ROI varia com o gênero? Kruskal-Wallis, Mann-Whitney e bootstrap."""
    v = filmes_com_retorno(df)
    v = v[v["Major Genre"] != "Não informado"]
    grupos = {g: s["ROI"].values for g, s in v.groupby("Major Genre", observed=True) if len(s) >= 30}
    kw = stats.kruskal(*grupos.values())

    terror, drama = grupos["Horror"], grupos["Drama"]
    mw = stats.mannwhitneyu(terror, drama, alternative="two-sided")
    rng = np.random.default_rng(SEMENTE)
    difs = [np.median(rng.choice(terror, len(terror))) - np.median(rng.choice(drama, len(drama)))
            for _ in range(5000)]
    ic = np.percentile(difs, [2.5, 97.5])

    imprimir("\n=== H4: retorno (ROI) por gênero ===")
    imprimir(f"Kruskal-Wallis ({len(grupos)} gêneros): H = {kw.statistic:.1f}, p {formatar_p(kw.pvalue)}")
    imprimir(f"ROI mediano: terror {np.median(terror):.2f}x (n = {len(terror)}), "
             f"drama {np.median(drama):.2f}x (n = {len(drama)})")
    imprimir(f"Filmes lucrativos: terror {(terror > 1).mean():.1%}, drama {(drama > 1).mean():.1%}")
    imprimir(f"Mann-Whitney (terror x drama): p {formatar_p(mw.pvalue)}")
    imprimir(f"IC 95% (bootstrap) da diferença das medianas: {ic[0]:.2f} a {ic[1]:.2f}")

    imprimir("\nROI mediano por década (terror x drama):")
    tabela_decada = roi_por_decada(df)
    imprimir(tabela_decada.to_string())
    return {"kw": kw, "mw": mw, "ic": ic, "decada": tabela_decada}


def roi_por_genero(df: pd.DataFrame, minimo: int = 30) -> pd.DataFrame:
    """ROI mediano, % de filmes lucrativos e nº de filmes por gênero (mínimo de filmes por gênero)."""
    v = filmes_com_retorno(df)
    tabela = v.groupby("Major Genre", observed=True).agg(
        filmes=("ROI", "size"),
        roi_mediano=("ROI", "median"),
        pct_lucrativos=("Lucro", lambda s: (s > 0).mean() * 100),
    )
    tabela = tabela[(tabela["filmes"] >= minimo) & (tabela.index != "Não informado")]
    return tabela.sort_values("roi_mediano", ascending=False).round(2)


def roi_por_decada(df: pd.DataFrame) -> pd.DataFrame:
    """ROI mediano de terror e drama nas décadas de 1980, 1990 e 2000."""
    v = filmes_com_retorno(df)
    v = v[v["Major Genre"].isin(["Horror", "Drama"]) & v["Decada"].isin([1980, 1990, 2000])]
    tabela = v.pivot_table(index="Decada", columns="Major Genre", values="ROI",
                           aggfunc="median", observed=True).round(2)
    tabela.columns = ["Drama" if c == "Drama" else "Terror" for c in tabela.columns]
    return tabela[["Terror", "Drama"]]


# ----------------------------------------------------------------------------
# 5. Gráficos
# ----------------------------------------------------------------------------
def salvar(figura, nome: str) -> None:
    """Salva a figura atual na pasta de gráficos."""
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    figura.tight_layout()
    figura.savefig(os.path.join(PASTA_GRAFICOS, nome), dpi=200)
    plt.close(figura)


def grafico_h1(df: pd.DataFrame, r: dict) -> None:
    """Figura 1: dispersão do orçamento contra a bilheteria (escala log) com reta de regressão."""
    v = filmes_com_retorno(df)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(v["Production Budget"], v["Worldwide Gross"], alpha=0.3, s=18, color=AZUL)
    x = np.logspace(np.log10(v["Production Budget"].min()), np.log10(v["Production Budget"].max()), 100)
    ax.plot(x, 10 ** (r["intercepto"] + r["inclinacao"] * np.log10(x)), color=VERMELHO,
            label="Reta de regressão (escala log)")
    ax.plot(x, x, color="gray", linestyle="--", label="Bilheteria = orçamento")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(rotulo_dolar))
    ax.yaxis.set_major_formatter(FuncFormatter(rotulo_dolar))
    ax.set_title(f"H1: orçamento x bilheteria mundial (Spearman = {r['rho']:.2f})")
    ax.set_xlabel("Orçamento de produção")
    ax.set_ylabel("Bilheteria mundial")
    ax.legend(loc="upper left")
    salvar(fig, "fig1_h1_orcamento_bilheteria.png")


def grafico_h2(df: pd.DataFrame, r: dict) -> None:
    """Figura 2: dispersão entre a nota da crítica e a nota do público."""
    par = df[["Rotten Tomatoes Rating", "IMDB 0-100"]].dropna()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.regplot(data=par, x="Rotten Tomatoes Rating", y="IMDB 0-100",
                scatter_kws={"alpha": 0.25, "s": 18, "color": AZUL},
                line_kws={"color": VERMELHO}, ax=ax)
    ax.set_title(f"H2: nota da crítica x nota do público (Spearman = {r['rho']:.2f})")
    ax.set_xlabel("Nota da crítica (Rotten Tomatoes, 0 a 100)")
    ax.set_ylabel("Nota do público (IMDB, escala 0 a 100)")
    salvar(fig, "fig2_h2_critica_publico.png")


def grafico_h3(df: pd.DataFrame, r: dict) -> None:
    """Figura 3: nota do público contra a bilheteria e contra o retorno (ROI)."""
    v = filmes_com_retorno(df).dropna(subset=["IMDB Rating"])
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, alvo, rotulo, chave in [
        (eixos[0], "Worldwide Gross", "Bilheteria mundial", "bilheteria"),
        (eixos[1], "ROI", "ROI (bilheteria ÷ orçamento)", "ROI"),
    ]:
        ax.scatter(v["IMDB Rating"], v[alvo], alpha=0.3, s=18, color=AZUL)
        ax.set_yscale("log")
        rho = r[("IMDB", chave)][0]
        ax.set_title(f"Nota x {chave} (Spearman = {rho:.2f})", fontsize=13)
        ax.set_xlabel("Nota do público (IMDB, 0 a 10)", fontsize=12)
        ax.set_ylabel(rotulo, fontsize=12)
        ax.tick_params(labelsize=11)
    eixos[0].yaxis.set_major_formatter(FuncFormatter(rotulo_dolar))
    eixos[1].axhline(1, color=VERMELHO, linestyle="--")
    fig.suptitle("H3: nota do público x bilheteria e x retorno", fontsize=15)
    salvar(fig, "fig3_h3_notas_bilheteria_roi.png")


def grafico_h4_genero(df: pd.DataFrame) -> None:
    """Figura 4: ROI mediano por gênero (barras horizontais, com nº de filmes)."""
    tabela = roi_por_genero(df).reset_index()
    tabela["Gênero"] = [f"{TRADUCAO_GENEROS[g]} (n={n})"
                        for g, n in zip(tabela["Major Genre"], tabela["filmes"])]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.barplot(data=tabela, y="Gênero", x="roi_mediano", color=AZUL, ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.2fx", padding=3)
    ax.axvline(1, color=VERMELHO, linestyle="--")
    ax.set_xlim(0, 3.3)
    ax.set_title("H4: ROI mediano (bilheteria ÷ orçamento) por gênero")
    ax.set_xlabel("ROI mediano (vezes). Linha tracejada = 1x (empate)")
    ax.set_ylabel("")
    salvar(fig, "fig4_h4_roi_genero.png")


def grafico_h4_decada(df: pd.DataFrame) -> None:
    """Figura 5: ROI mediano de terror e drama por década."""
    tabela = roi_por_decada(df).reset_index().melt(id_vars="Decada", var_name="Gênero", value_name="ROI mediano")
    tabela["Década"] = tabela["Decada"].astype(str) + "s"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.barplot(data=tabela, x="Década", y="ROI mediano", hue="Gênero",
                palette={"Terror": VERMELHO, "Drama": AZUL}, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2fx", padding=3)
    ax.axhline(1, color="gray", linestyle="--")
    ax.set_title("H4: terror x drama ao longo das décadas")
    ax.set_ylabel("ROI mediano (vezes)")
    salvar(fig, "fig5_h4_roi_decada.png")


def grafico_heatmap(df: pd.DataFrame) -> None:
    """Figura 6: mapa de calor das correlações de Spearman entre as variáveis numéricas."""
    nomes = {"Production Budget": "Orçamento", "Worldwide Gross": "Bilheteria", "ROI": "ROI",
             "IMDB Rating": "Nota IMDB", "Rotten Tomatoes Rating": "Nota crítica",
             "IMDB Votes": "Votos IMDB", "Ano": "Ano"}
    corr = filmes_com_retorno(df)[list(nomes)].rename(columns=nomes).corr(method="spearman")
    mascara = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, mask=mascara, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Correlações de Spearman entre as variáveis numéricas")
    salvar(fig, "fig6_heatmap_correlacoes.png")


# ----------------------------------------------------------------------------
# 6. Execução
# ----------------------------------------------------------------------------
def main() -> None:
    baixar_dados_se_necessario(CAMINHO_JSON, URL_DADOS)
    bruto = carregar_dados(CAMINHO_JSON)
    inspecionar_dados(bruto)

    df = tratar_dados(bruto)
    imprimir("\nAusentes após o tratamento (só colunas com ausentes):")
    ausentes = df.isna().sum()
    imprimir(ausentes[ausentes > 0].sort_values(ascending=False).to_string())
    estatisticas_descritivas(df)

    r1, r2, r3, r4 = teste_h1(df), teste_h2(df), teste_h3(df), teste_h4(df)

    grafico_h1(df, r1)
    grafico_h2(df, r2)
    grafico_h3(df, r3)
    grafico_h4_genero(df)
    grafico_h4_decada(df)
    grafico_heatmap(df)

    df.to_csv("filmes_tratados.csv", index=False)
    with open("resultados.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(LINHAS_RESULTADO))
    imprimir(f"\nGráficos salvos na pasta '{PASTA_GRAFICOS}/'. Resultados em 'resultados.txt'.")


if __name__ == "__main__":
    main()
