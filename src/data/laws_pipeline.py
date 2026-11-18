"""OAB law corpus: select the registry statutes, clean them, chunk by legal structure.

    python -m src.data.laws_pipeline [--with-supplementary] [--target-year YEAR]

Output: ``data/laws/laws_v*.csv``.
"""

import json
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import tiktoken
from loguru import logger

# Logger
LOG_FILE = Path("logs/laws_pipeline.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level:<8}</level> | {function:<24} | {message}"
)
logger.remove()
logger.add(sys.stdout, format=LOG_FORMAT, colorize=True)
logger.add(LOG_FILE, format=LOG_FORMAT, rotation="10 MB", retention=5,
           level="INFO", enqueue=True)

# Configuration
DATA_PATH = Path("data/laws")
CSV_FILENAME = "legislaai/compilado_tabelas___modificado_2026-03-05T19_24_51.651368849Z.csv"
EXPORT_FILENAME = "laws_v1.csv"
EXPORT_FILENAME_V2 = "laws_v2.csv"
# A resolved corpus fixes each article to the wording in force at a target year.
RESOLVED_VERSION = "v5"
EXPORT_TEMPLATE_RESOLVED = "laws_{version}_{target_year}.csv"
SUPPLEMENTARY_DIR = "supplementary"
SUPPLEMENTARY_SUMMARY = "summary.json"
# First day in force of acts with a vacatio legis (hand-curated).
ENTRY_INTO_FORCE = "entry_into_force.json"

# Sentinel for "use the pipeline's target_year"; an explicit None means unresolved.
_INHERIT_TARGET_YEAR = object()

ENCODING = tiktoken.get_encoding("cl100k_base")

# Chunk sizing (tokens): TARGET is the soft goal, MAX only to keep an article whole.
CHUNK_TARGET_TOKENS = 1200
CHUNK_MAX_TOKENS = 2600
CHUNK_MIN_TOKENS = 300

_PT_MONTHS = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}

CONST_1988_MARKER = "CONSTITUICAO DA REPUBLICA FEDERATIVA DO BRASIL DE 1988"

# OAB law registry: (num, source_bucket, true_tipo, label, area).
# Pre-1988 codes live only in the raw data's mislabeled "Código Civil" bucket.
REGISTRY: List[Tuple[int, str, str, str, str]] = [
    # Direito Civil
    (10406, "Código Civil", "Lei Ordinária", "Código Civil", "Civil"),
    (9610,  "Lei Ordinária", "Lei Ordinária", "Lei de Direitos Autorais", "Civil"),
    (12965, "Lei Ordinária", "Lei Ordinária", "Marco Civil da Internet", "Civil"),
    (13709, "Lei Ordinária", "Lei Ordinária", "LGPD — Lei Geral de Proteção de Dados", "Civil"),
    (8245,  "Lei Ordinária", "Lei Ordinária", "Lei do Inquilinato", "Civil"),
    (9307,  "Lei Ordinária", "Lei Ordinária", "Lei de Arbitragem", "Civil"),
    (13140, "Lei Ordinária", "Lei Ordinária", "Lei de Mediação", "Civil"),
    (9514,  "Lei Ordinária", "Lei Ordinária", "Lei da Alienação Fiduciária de Imóveis", "Civil"),
    (8009,  "Lei Ordinária", "Lei Ordinária", "Lei do Bem de Família", "Civil"),
    (11804, "Lei Ordinária", "Lei Ordinária", "Lei de Alimentos Gravídicos", "Civil"),
    (12318, "Lei Ordinária", "Lei Ordinária", "Lei da Alienação Parental", "Civil"),
    (8560,  "Lei Ordinária", "Lei Ordinária", "Lei de Investigação de Paternidade", "Civil"),
    # Direito Processual Civil
    (13105, "Lei Ordinária", "Lei Ordinária", "Código de Processo Civil (CPC/2015)", "Processo Civil"),
    (9099,  "Lei Ordinária", "Lei Ordinária", "Lei dos Juizados Especiais", "Processo Civil"),
    (12016, "Lei Ordinária", "Lei Ordinária", "Lei do Mandado de Segurança", "Processo Civil"),
    (10259, "Lei Ordinária", "Lei Ordinária", "Lei dos Juizados Especiais Federais", "Processo Civil"),
    (9507,  "Lei Ordinária", "Lei Ordinária", "Lei do Habeas Data", "Processo Civil"),
    (11419, "Lei Ordinária", "Lei Ordinária", "Lei do Processo Eletrônico", "Processo Civil"),
    (12153, "Lei Ordinária", "Lei Ordinária", "Lei dos Juizados Especiais da Fazenda Pública", "Processo Civil"),
    (9494,  "Lei Ordinária", "Lei Ordinária", "Lei da Tutela Antecipada contra a Fazenda Pública", "Processo Civil"),
    (8437,  "Lei Ordinária", "Lei Ordinária", "Lei das Medidas Cautelares contra o Poder Público", "Processo Civil"),
    # Direito Constitucional
    (9868,  "Lei Ordinária", "Lei Ordinária", "Lei da ADI e ADC", "Constitucional"),
    (9882,  "Lei Ordinária", "Lei Ordinária", "Lei da ADPF", "Constitucional"),
    (13300, "Lei Ordinária", "Lei Ordinária", "Lei do Mandado de Injunção", "Constitucional"),
    (9709,  "Lei Ordinária", "Lei Ordinária", "Lei do Plebiscito, Referendo e Iniciativa Popular", "Constitucional"),
    (11417, "Lei Ordinária", "Lei Ordinária", "Lei da Súmula Vinculante (regulamenta art. 103-A CF)", "Constitucional"),
    # Direito Empresarial
    (11101, "Lei Ordinária", "Lei Ordinária", "Lei de Recuperação e Falências", "Empresarial"),
    (14112, "Lei Ordinária", "Lei Ordinária", "Reforma da Lei de Falências (14.112/2020)", "Empresarial"),
    (9279,  "Lei Ordinária", "Lei Ordinária", "Lei da Propriedade Industrial", "Empresarial"),
    (8934,  "Lei Ordinária", "Lei Ordinária", "Lei de Registro Público de Empresas Mercantis", "Empresarial"),
    (12529, "Lei Ordinária", "Lei Ordinária", "Lei de Defesa da Concorrência (CADE)", "Empresarial"),
    (9492,  "Lei Ordinária", "Lei Ordinária", "Lei do Protesto de Títulos", "Empresarial"),
    (13966, "Lei Ordinária", "Lei Ordinária", "Nova Lei de Franquia (13.966/2019)", "Empresarial"),
    # Direito Penal
    (2848,  "Código Civil", "Decreto-Lei", "Código Penal", "Penal"),
    (11343, "Lei Ordinária", "Lei Ordinária", "Lei de Drogas", "Penal"),
    (8072,  "Lei Ordinária", "Lei Ordinária", "Lei dos Crimes Hediondos", "Penal"),
    (12850, "Lei Ordinária", "Lei Ordinária", "Lei das Organizações Criminosas", "Penal"),
    (13869, "Lei Ordinária", "Lei Ordinária", "Lei de Abuso de Autoridade", "Penal"),
    (13964, "Lei Ordinária", "Lei Ordinária", "Pacote Anticrime (13.964/2019)", "Penal"),
    (9455,  "Lei Ordinária", "Lei Ordinária", "Lei da Tortura", "Penal"),
    (9613,  "Lei Ordinária", "Lei Ordinária", "Lei de Lavagem de Dinheiro", "Penal"),
    (8137,  "Lei Ordinária", "Lei Ordinária", "Lei dos Crimes contra a Ordem Tributária", "Penal"),
    (9503,  "Código Civil", "Lei Ordinária", "Código de Trânsito Brasileiro (CTB)", "Penal"),
    (7716,  "Lei Ordinária", "Lei Ordinária", "Lei do Racismo", "Penal"),
    (13260, "Lei Ordinária", "Lei Ordinária", "Lei Antiterrorismo", "Penal"),
    # Direito Processual Penal
    (3689,  "Código Civil", "Decreto-Lei", "Código de Processo Penal (CPP)", "Processo Penal"),
    (9296,  "Lei Ordinária", "Lei Ordinária", "Lei da Interceptação Telefônica", "Processo Penal"),
    (7960,  "Lei Ordinária", "Lei Ordinária", "Lei da Prisão Temporária", "Processo Penal"),
    (12037, "Lei Ordinária", "Lei Ordinária", "Lei da Identificação Criminal", "Processo Penal"),
    (12830, "Lei Ordinária", "Lei Ordinária", "Lei da Investigação Criminal pelo Delegado de Polícia", "Processo Penal"),
    # Direito do Trabalho
    (5452,  "Código Civil", "Decreto-Lei", "CLT — Consolidação das Leis do Trabalho", "Trabalho"),
    (13467, "Lei Ordinária", "Lei Ordinária", "Reforma Trabalhista (13.467/2017)", "Trabalho"),
    (8036,  "Lei Ordinária", "Lei Ordinária", "Lei do FGTS", "Trabalho"),
    (150,   "Lei Complementar", "Lei Complementar", "Lei do Empregado Doméstico (LC 150/2015)", "Trabalho"),
    (7783,  "Lei Ordinária", "Lei Ordinária", "Lei de Greve", "Trabalho"),
    (12506, "Lei Ordinária", "Lei Ordinária", "Lei do Aviso Prévio Proporcional", "Trabalho"),
    (10101, "Lei Ordinária", "Lei Ordinária", "Lei da Participação nos Lucros ou Resultados (PLR)", "Trabalho"),
    (9029,  "Lei Ordinária", "Lei Ordinária", "Lei de Práticas Discriminatórias na Relação de Trabalho", "Trabalho"),
    (7998,  "Lei Ordinária", "Lei Ordinária", "Lei do Seguro-Desemprego e Abono Salarial", "Trabalho"),
    (11788, "Lei Ordinária", "Lei Ordinária", "Lei do Estágio", "Trabalho"),
    (12690, "Lei Ordinária", "Lei Ordinária", "Lei das Cooperativas de Trabalho", "Trabalho"),
    (9601,  "Lei Ordinária", "Lei Ordinária", "Lei do Contrato de Trabalho por Prazo Determinado", "Trabalho"),
    # Direito Processual do Trabalho
    (7701,  "Lei Ordinária", "Lei Ordinária", "Lei de Organização da Justiça do Trabalho (TST)", "Processo do Trabalho"),
    # Direito Administrativo
    (14133, "Lei Ordinária", "Lei Ordinária", "Nova Lei de Licitações e Contratos", "Administrativo"),
    (8666,  "Lei Ordinária", "Lei Ordinária", "Lei de Licitações (8.666/1993)", "Administrativo"),
    (8429,  "Lei Ordinária", "Lei Ordinária", "Lei de Improbidade Administrativa", "Administrativo"),
    (14230, "Lei Ordinária", "Lei Ordinária", "Reforma da Lei de Improbidade (14.230/2021)", "Administrativo"),
    (9784,  "Lei Ordinária", "Lei Ordinária", "Lei do Processo Administrativo Federal", "Administrativo"),
    (8112,  "Lei Ordinária", "Lei Ordinária", "Estatuto dos Servidores Públicos Federais", "Administrativo"),
    (12527, "Lei Ordinária", "Lei Ordinária", "Lei de Acesso à Informação", "Administrativo"),
    (8987,  "Lei Ordinária", "Lei Ordinária", "Lei de Concessões e Permissões", "Administrativo"),
    (10257, "Lei Ordinária", "Lei Ordinária", "Estatuto da Cidade", "Administrativo"),
    (12846, "Lei Ordinária", "Lei Ordinária", "Lei Anticorrupção", "Administrativo"),
    (11107, "Lei Ordinária", "Lei Ordinária", "Lei dos Consórcios Públicos", "Administrativo"),
    (13303, "Lei Ordinária", "Lei Ordinária", "Lei Geral das Estatais", "Administrativo"),
    (13848, "Lei Ordinária", "Lei Ordinária", "Lei das Agências Reguladoras", "Administrativo"),
    (13874, "Lei Ordinária", "Lei Ordinária", "Lei da Liberdade Econômica", "Administrativo"),
    (9637,  "Lei Ordinária", "Lei Ordinária", "Lei das Organizações Sociais", "Administrativo"),
    (9790,  "Lei Ordinária", "Lei Ordinária", "Lei das OSCIP", "Administrativo"),
    (13019, "Lei Ordinária", "Lei Ordinária", "Marco Regulatório das Organizações da Sociedade Civil (MROSC)", "Administrativo"),
    (12462, "Lei Ordinária", "Lei Ordinária", "Regime Diferenciado de Contratações Públicas (RDC)", "Administrativo"),
    (13460, "Lei Ordinária", "Lei Ordinária", "Lei dos Direitos dos Usuários de Serviços Públicos", "Administrativo"),
    (9074,  "Lei Ordinária", "Lei Ordinária", "Lei de Concessões e Permissões Complementares", "Administrativo"),
    (8745,  "Lei Ordinária", "Lei Ordinária", "Lei da Contratação Temporária no Serviço Público Federal", "Administrativo"),
    (12587, "Lei Ordinária", "Lei Ordinária", "Política Nacional de Mobilidade Urbana", "Administrativo"),
    (9636,  "Lei Ordinária", "Lei Ordinária", "Lei dos Bens Imóveis da União", "Administrativo"),
    # Direito Tributário
    (5172,  "Código Civil", "Lei Ordinária", "Código Tributário Nacional (CTN)", "Tributário"),
    (105,   "Lei Complementar", "Lei Complementar", "Lei do Sigilo Bancário (LC 105/2001)", "Tributário"),
    (87,    "Lei Complementar", "Lei Complementar", "Lei Kandir — ICMS (LC 87/1996)", "Tributário"),
    (116,   "Lei Complementar", "Lei Complementar", "Lei do ISS (LC 116/2003)", "Tributário"),
    (24,    "Lei Complementar", "Lei Complementar", "LC 24/1975 — Convênios do ICMS (Isenções)", "Tributário"),
    (214,   "Lei Complementar", "Lei Complementar", "LC 214/2025 — IBS, CBS e Imposto Seletivo (Reforma Tributária)", "Tributário"),
    # Emendas Constitucionais
    # Transition rules that live only in the amendment text.
    (45,    "Emenda Constitucional", "Emenda Constitucional", "EC 45/2004 — Reforma do Judiciário", "Constitucional"),
    (103,   "Emenda Constitucional", "Emenda Constitucional", "EC 103/2019 — Reforma da Previdência", "Previdenciário"),
    (132,   "Emenda Constitucional", "Emenda Constitucional", "EC 132/2023 — Reforma Tributária", "Tributário"),
    # Direito Financeiro
    (101,   "Lei Complementar", "Lei Complementar", "Lei de Responsabilidade Fiscal (LRF)", "Financeiro"),
    # Direito do Consumidor
    (8078,  "Lei Ordinária", "Lei Ordinária", "Código de Defesa do Consumidor (CDC)", "Consumidor"),
    # Direitos Humanos / Vulneráveis
    (8069,  "Lei Ordinária", "Lei Ordinária", "Estatuto da Criança e do Adolescente (ECA)", "Direitos Humanos"),
    (10741, "Lei Ordinária", "Lei Ordinária", "Estatuto da Pessoa Idosa", "Direitos Humanos"),
    (13146, "Lei Ordinária", "Lei Ordinária", "Estatuto da Pessoa com Deficiência (LBI)", "Direitos Humanos"),
    (12288, "Lei Ordinária", "Lei Ordinária", "Estatuto da Igualdade Racial", "Direitos Humanos"),
    (12594, "Lei Ordinária", "Lei Ordinária", "SINASE — Sistema Nacional de Atendimento Socioeducativo", "Direitos Humanos"),
    (12852, "Lei Ordinária", "Lei Ordinária", "Estatuto da Juventude", "Direitos Humanos"),
    (13431, "Lei Ordinária", "Lei Ordinária", "Lei da Escuta Especializada e Depoimento Especial", "Direitos Humanos"),
    (12986, "Lei Ordinária", "Lei Ordinária", "Lei do Conselho Nacional dos Direitos Humanos", "Direitos Humanos"),
    # Ética Profissional
    (8906,  "Lei Ordinária", "Lei Ordinária", "Estatuto da Advocacia e da OAB", "Ética"),
    # Direito Eleitoral
    (9504,  "Lei Ordinária", "Lei Ordinária", "Lei das Eleições", "Eleitoral"),
    (9096,  "Lei Ordinária", "Lei Ordinária", "Lei dos Partidos Políticos", "Eleitoral"),
    (4737,  "Código Civil", "Lei Ordinária", "Código Eleitoral", "Eleitoral"),
    (135,   "Lei Complementar", "Lei Complementar", "Lei da Ficha Limpa (LC 135/2010)", "Eleitoral"),
    (64,    "Lei Complementar", "Lei Complementar", "Lei das Inelegibilidades (LC 64/1990)", "Eleitoral"),
    # Direito Previdenciário
    (8213,  "Lei Ordinária", "Lei Ordinária", "Lei de Benefícios da Previdência (RGPS)", "Previdenciário"),
    (8212,  "Lei Ordinária", "Lei Ordinária", "Lei de Custeio da Seguridade Social", "Previdenciário"),
    (8742,  "Lei Ordinária", "Lei Ordinária", "LOAS — Lei Orgânica da Assistência Social", "Previdenciário"),
    (108,   "Lei Complementar", "Lei Complementar", "LC 108/2001 — Relação Poder Público / Entidades Fechadas de Previdência Complementar", "Previdenciário"),
    (109,   "Lei Complementar", "Lei Complementar", "LC 109/2001 — Regime de Previdência Complementar", "Previdenciário"),
    # Direito Empresarial (tributário / MPE)
    (123,   "Lei Complementar", "Lei Complementar", "Estatuto da Micro e Pequena Empresa (Simples)", "Empresarial"),
    # Direito Internacional
    (13445, "Lei Ordinária", "Lei Ordinária", "Lei de Migração", "Internacional"),
    (9474,  "Lei Ordinária", "Lei Ordinária", "Lei do Refúgio (Estatuto dos Refugiados)", "Internacional"),
    # Direito Ambiental
    (9605,  "Lei Ordinária", "Lei Ordinária", "Lei de Crimes Ambientais", "Ambiental"),
    (12651, "Lei Ordinária", "Lei Ordinária", "Código Florestal", "Ambiental"),
    (9985,  "Lei Ordinária", "Lei Ordinária", "Lei do SNUC (Unidades de Conservação)", "Ambiental"),
    (12305, "Lei Ordinária", "Lei Ordinária", "Política Nacional de Resíduos Sólidos", "Ambiental"),
    (11105, "Lei Ordinária", "Lei Ordinária", "Lei de Biossegurança", "Ambiental"),
    (140,   "Lei Complementar", "Lei Complementar", "LC 140/2011 — Competência Administrativa Ambiental", "Ambiental"),
    (9433,  "Lei Ordinária", "Lei Ordinária", "Política Nacional de Recursos Hídricos", "Ambiental"),
    (11445, "Lei Ordinária", "Lei Ordinária", "Lei de Saneamento Básico", "Ambiental"),
    (11428, "Lei Ordinária", "Lei Ordinária", "Lei da Mata Atlântica", "Ambiental"),
    (9795,  "Lei Ordinária", "Lei Ordinária", "Política Nacional de Educação Ambiental", "Ambiental"),
    (12187, "Lei Ordinária", "Lei Ordinária", "Política Nacional sobre Mudança do Clima", "Ambiental"),
    (11284, "Lei Ordinária", "Lei Ordinária", "Lei de Gestão de Florestas Públicas", "Ambiental"),
    (7661,  "Lei Ordinária", "Lei Ordinária", "Plano Nacional de Gerenciamento Costeiro", "Ambiental"),
    # Direito à Educação (transversal, alta incidência)
    (9394,  "Lei Ordinária", "Lei Ordinária", "LDB — Diretrizes e Bases da Educação", "Administrativo"),
    (12711, "Lei Ordinária", "Lei Ordinária", "Lei de Cotas", "Administrativo"),
    (10826, "Lei Ordinária", "Lei Ordinária", "Estatuto do Desarmamento", "Penal"),
]

# Last day or year in force of registry acts repealed as a whole by another act.
REGISTRY_VALID_UNTIL: Dict[Tuple[str, int], Union[int, str]] = {
    # Revoked on 30 Dec 2023 (Lei 14.133/2021 art. 193, II, as amended by LC 198/2023).
    ("Lei Ordinária", 8666): "2023-12-29",
}


# OAB statutes absent from the LegislaAI scrape (it covers ordinary laws from 1988 on).
KNOWN_ABSENT = {
    7210: "Lei de Execução Penal (LEP, 1984)",
    11340: "Lei Maria da Penha (2006)",
    4657: "LINDB (Decreto-Lei 4.657/1942)",
    4320: "Normas Gerais de Direito Financeiro (1964)",
    6830: "Lei de Execução Fiscal (1980)",
    6015: "Lei de Registros Públicos (1973)",
    4717: "Lei da Ação Popular (1965)",
    7347: "Lei da Ação Civil Pública (1985)",
    6404: "Lei das Sociedades Anônimas (1976)",
    200: "Decreto-Lei 200/1967 (Organização Administrativa)",
    3365: "Decreto-Lei 3.365/1941 (Desapropriação)",
    3688: "Lei das Contravenções Penais (1941)",
    11079: "Lei das Parcerias Público-Privadas (PPP, 2004)",
    70235: "Decreto 70.235/1972 (Processo Administrativo Fiscal, PAF) — pre-1988 decreto, not in the 'Decreto' tipo bucket (which only covers 2025)",
    6938: "Lei da Política Nacional do Meio Ambiente (PNMA, 1981)",
    5478: "Lei de Alimentos (1968)",
    57663: "Decreto 57.663/1966 (Lei Uniforme de Genebra sobre títulos de crédito)",
    7357: "Lei do Cheque (1985)",
    5474: "Lei da Duplicata (1968)",
    5584: "Lei 5.584/1970 (rito sumaríssimo no processo do trabalho)",
    3048: "Decreto 3.048/1999 (Regulamento da Previdência Social) — also out of scope by design: a regulatory decree, not primary legislation",
    4898: "Lei 4.898/1965 (antiga Lei de Abuso de Autoridade, revogada pela Lei 13.869/2019)",
    6815: "Lei 6.815/1980 (antigo Estatuto do Estrangeiro, revogado pela Lei de Migração 13.445/2017)",
    6019: "Lei 6.019/1974 (Trabalho Temporário / Terceirização)",
    605: "Lei 605/1949 (Repouso Semanal Remunerado)",
    4090: "Lei 4.090/1962 (13º Salário)",
    4749: "Lei 4.749/1965 (13º Salário, regulamentação)",
    7418: "Lei 7.418/1985 (Vale-Transporte)",
    6766: "Lei 6.766/1979 (Parcelamento do Solo Urbano)",
    6091: "Lei 6.091/1974 (Transporte Gratuito de Eleitores)",
    7492: "Lei 7.492/1986 (Crimes contra o Sistema Financeiro Nacional)",
    2044: "Decreto 2.044/1908 (Letra de Câmbio e Nota Promissória)",
    1079: "Lei 1.079/1950 (Crimes de Responsabilidade)",
    2889: "Lei 2.889/1956 (Genocídio)",
    4591: "Lei 4.591/1964 (Condomínios e Incorporações Imobiliárias)",
    7064: "Lei 7.064/1982 (Trabalhadores Contratados/Transferidos para o Exterior)",
    6385: "Lei 6.385/1976 (CVM — Comissão de Valores Mobiliários)",
    4728: "Lei 4.728/1965 (Mercado de Capitais)",
    1973: "Decreto 1.973/1996 (Convenção de Belém do Pará)",
    4377: "Decreto 4.377/2002 (CEDAW)",
    99710: "Decreto 99.710/1990 (Convenção sobre os Direitos da Criança)",
    65810: "Decreto 65.810/1969 (Convenção Internacional sobre Eliminação da Discriminação Racial)",
    18871: "Decreto 18.871/1929 (Código Bustamante)",
    7030: "Decreto 7.030/2009 (Convenção de Viena sobre o Direito dos Tratados)",
    56435: "Decreto 56.435/1965 (Convenção de Viena sobre Relações Diplomáticas)",
    61078: "Decreto 61.078/1967 (Convenção de Viena sobre Relações Consulares)",
    4311: "Decreto 4.311/2002 (Convenção de Nova York sobre Reconhecimento de Sentenças Arbitrais Estrangeiras)",
}

# Tested content that is not federal legislation, so no scrape covers it.
KNOWN_ABSENT_NON_LEGISLATIVE = [
    "Código de Ética e Disciplina da OAB (ato normativo do Conselho Federal da OAB, não é lei federal)",
    "Regulamento Geral da OAB (idem — ato normativo do CFOAB)",
    "Provimentos do Conselho Federal da OAB (atos normativos infralegais do CFOAB, ex.: "
    "regulamentação de exame de ordem, sociedade de advogados, publicidade) — mesma natureza "
    "do CED/Regulamento Geral, não é lei federal",
    "Súmulas do Conselho Federal da OAB (súmulas administrativas/normativas do próprio "
    "conselho, distintas das súmulas judiciais do STF/STJ/TST já listadas abaixo — também "
    "não são legislação federal, e o dataset bruto só contém leis, não atos de conselhos "
    "profissionais)",
    "Filosofia do Direito — não corresponde a nenhuma lei; é conteúdo doutrinário, "
    "estruturalmente fora do escopo de um corpus legislativo",
    "Súmulas Vinculantes (STF), Súmulas do STJ/TST, Orientações Jurisprudenciais (OJs) — "
    "jurisprudência, não legislação; tipo de conteúdo ausente do dataset bruto inteiro",
    "Tratados de Direitos Humanos incorporados por decreto de promulgação — CADH/Pacto de "
    "San José (Decreto 678/1992), PIDCP (Decreto 592/1992), PIDESC (Decreto 591/1992), "
    "Convenção contra a Tortura (Decreto 40/1991) — confirmados ausentes do dataset bruto "
    "(checados diretamente, ver KNOWN_ABSENT para outros tratados clássicos igualmente "
    "ausentes: Belém do Pará, CEDAW, Convenção da Criança, Discriminação Racial, Bustamante, "
    "Viena/Tratados, Viena/Diplomáticas, Viena/Consulares, Nova York/Arbitragem). A bucket "
    "'Decreto-Emenda' tem só 6 linhas (racismo, deficiência/PCD, Marraqueche — ver "
    "KNOWN_FRAGMENTARY), e nenhuma delas é utilizável mesmo estando 'presente'",
    "Declaração Universal dos Direitos Humanos (DUDH) — declaração da ONU, não incorporada "
    "por nenhum decreto brasileiro específico com força de lei; puramente declaratória, sem "
    "num_lei para checar no dataset",
    "Resoluções do Conselho Federal da OAB — outro tipo de ato normativo infralegal do "
    "CFOAB, mesma natureza dos Provimentos/CED/Regulamento Geral acima",
    "Instruções Normativas da OAB — idem",
    "Manual de Procedimentos do Processo Ético-Disciplinar da OAB — material institucional, "
    "não legislação federal",
    "Resoluções do TSE — normas eleitorais infralegais do Tribunal Superior Eleitoral, não "
    "leis federais",
    "Resoluções do CNJ — normas administrativas do Conselho Nacional de Justiça, não leis "
    "federais",
    "Jurisprudência da Corte Interamericana de Direitos Humanos (Corte IDH) — jurisprudência "
    "internacional, não legislação",
    "Decretos regulamentares em geral (ex.: Decreto 3.048/1999) são excluídos por critério "
    "de escopo — o REGISTRY foca em legislação primária (leis, leis complementares, "
    "decretos-lei historicamente equiparados a lei); 'Decreto-Emenda' (promulgação de "
    "tratado) é avaliado caso a caso quando disponível",
]

# Fragmentary scrape rows replaced by a hand-sourced supplementary file.
SUPPLEMENTARY_OVERRIDES: Dict[Tuple[str, int], str] = {
    ("Lei Ordinária", 8666): "scrape keeps 1 live article; Planalto text has all 126 with their revocation annotations",
    ("Lei Ordinária", 12462): "scrape starts at Art. 9; Planalto text has Arts. 1-8",
}

# Scraped acts with (near-)empty text: superseded laws or treaty redirect stubs.
KNOWN_FRAGMENTARY = {
    8666: "Lei de Licitações (8.666/1993) — 1 live article survives (Art. 3), rest revoked; full text sourced by hand (SUPPLEMENTARY_OVERRIDES)",
    12462: "RDC (12.462/2011) — scraped text starts at Art. 9; full text sourced by hand (SUPPLEMENTARY_OVERRIDES)",
    10520: "Lei do Pregão (10.520/2002) — 0 live articles, fully revoked by Lei 14.133/2021; not added",
    8955: "Lei de Franquia antiga (8.955/1994) — 0 live articles, fully revoked by Lei 13.966/2019; not added",
    6949: "Convenção sobre os Direitos das Pessoas com Deficiência (Decreto 6.949/2009) — row present in 'Decreto-Emenda' bucket but only a 196-char redirect stub, 0 articles; not added",
    9522: "Tratado de Marraqueche (Decreto 9.522/2018) — same stub pattern as 6.949, 0 articles; not added",
    10932: "Convenção Interamericana contra o Racismo (Decreto 10.932/2022) — same stub pattern, 0 articles; not added",
}


# Helpers

def count_tokens(text: Union[str, float]) -> int:
    if pd.isna(text):
        return 0
    return len(ENCODING.encode(str(text)))


def clean_num(val: Union[str, float]) -> Union[str, int]:
    if pd.isna(val) or str(val).strip().lower() in ("", "nan"):
        return ""
    s = str(val).strip()
    # Digits and separators only, so ids like '69e209c8' are not read as floats.
    digits = s.replace(".", "").replace(",", "")
    if digits.isdigit():
        try:
            return int(digits)
        except (ValueError, OverflowError):
            return s
    return s


def _parse_pt_date(raw: Union[str, float]) -> Optional[date]:
    """Parse the Legislatech Portuguese/ISO date formats."""
    if not raw or pd.isna(raw):
        return None
    raw = str(raw).strip().lower()
    m = re.match(r"(\w+)\s+(\d{1,2}),\s*(\d{4})", raw)          # "abril 9, 2000"
    if m and _PT_MONTHS.get(m.group(1)):
        try:
            return date(int(m.group(3)), _PT_MONTHS[m.group(1)], int(m.group(2)))
        except ValueError:
            pass
    m = re.match(r"(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})", raw)  # "9 de abril de 2000"
    if m and _PT_MONTHS.get(m.group(2)):
        try:
            return date(int(m.group(3)), _PT_MONTHS[m.group(2)], int(m.group(1)))
        except ValueError:
            pass
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)                # ISO
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def fmt_num(num: Union[int, str]) -> str:
    """10406 -> '10.406' (Brazilian thousands separator)."""
    try:
        return f"{int(num):,}".replace(",", ".")
    except (ValueError, TypeError):
        return str(num)


def display_tipo(tipo: str, num: Union[int, str], ano: Optional[int]) -> str:
    """Human citation label, e.g. 'Lei Complementar nº 101/2000'."""
    year = ano if ano else "?"
    prefix = {
        "Lei Ordinária": "Lei",
        "Lei Complementar": "Lei Complementar",
        "Decreto-Lei": "Decreto-Lei",
        "Constituição": "Constituição Federal",
    }.get(tipo, tipo)
    if tipo == "Constituição":
        return "Constituição Federal de 1988"
    # Acts with no number (CFOAB acts) are cited by year alone.
    if num is None or (isinstance(num, float) and pd.isna(num)) or str(num).strip() in ("", "nan", "None"):
        return f"{prefix} ({year})"
    return f"{prefix} nº {fmt_num(num)}/{year}"


# Cleaning

# Scraper boilerplate lines to drop wherever they occur.
_BOILERPLATE_LINE = re.compile(
    r"(?im)^\s*(?:"
    r"presid[êe]ncia da rep[úu]blica.*"
    r"|casa civil.*"
    r"|secretaria[- ]?geral.*"
    r"|subchefia para assuntos jur[íi]dicos.*"
    r"|secretaria especial para assuntos jur[íi]dicos.*"
    r"|texto compilado.*"
    r"|texto para impress[ãa]o.*"
    r"|multivig[êe]ncia.*"
    r"|[íi]ndice\s*"
    r"|vig[êe]ncia\s*"
    r"|mensagem de veto.*"
    r"|promulga[çc][ãa]o partes vetadas.*"
    r"|\(promulga[çc][ãa]o.*\)\s*"
    r"|download.*"
    r"|este texto n[ãa]o substitui.*"
    r"|\*+\s*"
    r")$"
)

# The enacting clause that separates the preamble from the operative text.
_ENACT = re.compile(
    r"(?is)(?:"
    r"sanciono a seguinte lei\s+complementar|sanciono a seguinte lei|"
    r"promulgo a seguinte lei\s+complementar|promulgo a seguinte lei|"
    r"decreta\s*:|decreto\s*:"
    r")"
)

# First structural anchor (fallback header cut when no enacting clause found).
_FIRST_ANCHOR = re.compile(
    r"(?im)^\s*(?:PARTE\s|LIVRO\s+[IVXLCDM]|T[ÍI]TULO\s+[IVXLCDM]|Art\.?\s*1\b)"
)

# Inline provenance annotations, optionally naming their provision ("(Inciso incluído …)").
_ANN_PROVISION = r"(?:(?:par[áa]grafo|inciso|al[íi]nea|caput|item)\s+)?"
_ANN_DROP = re.compile(
    rf"\((?:\s*(?:vide|reda[çc][ãa]o (?:dada|da pel|pel|restabelecid)|restabelecid[oa]|"
    rf"{_ANN_PROVISION}(?:inclu[íi]d[oa]|acrescentad[oa])|renumera(?:d[oa]|ndo)|"
    r"vig[êe]ncia|regulament[oa]|regulamenta[çc][ãa]o|produç[ãa]o de efeito|"
    r"revigorad[oa]|com reda[çc][ãa]o|vide adin?|vide adpf|vide adc|vide adi)"
    r")[^()]*\)",
    flags=re.IGNORECASE,
)
_ANN_REVOKED = re.compile(r"\(\s*revogad[oa][^()]*\)", flags=re.IGNORECASE)
# Same annotations with the closing paren lost by the scraper: strip to end of line.
_ANN_KEYWORDS = (rf"vide|{_ANN_PROVISION}(?:inclu[íi]d[oa]|acrescentad[oa])|"
                 r"reda[çc][ãa]o (?:dada|da pel|pel|restabelecid)|restabelecid[oa]|vig[êe]ncia|"
                 r"renumera(?:d[oa]|ndo)|regulament[oa]|regulamenta[çc][ãa]o|"
                 r"produ[çc][ãa]o de efeito|revigorad[oa]|com reda[çc][ãa]o")
_ANN_REVOKED_OPEN = re.compile(r"\(\s*revogad[oa][^)\n]*$", flags=re.IGNORECASE | re.MULTILINE)
_ANN_DROP_OPEN = re.compile(rf"\((?:{_ANN_KEYWORDS})[^)\n]*$", flags=re.IGNORECASE | re.MULTILINE)
_NR_MARKER = re.compile(r"\(\s*NR\s*\)")
_DOT_LEADERS = re.compile(r"\.{4,}")

_BRASILIA_SIGNOFF = re.compile(r"Bras[ií]lia\s*,.*?\d{4}", flags=re.IGNORECASE | re.DOTALL)

# Per-article most recent amendment year, read from annotations before they are stripped.
_AMEND_KEYWORDS = (rf"reda[çc][ãa]o dada|{_ANN_PROVISION}inclu[íi]d[oa]|renumerad[oa]|"
                   r"revogad[oa]|revigorad[oa]|com reda[çc][ãa]o")
_AMEND_ANNOTATION = re.compile(rf"\((?:{_AMEND_KEYWORDS})[^)\n]*", flags=re.IGNORECASE)
_YEAR_TOKEN = re.compile(r"\d{4}")
_ART_NUM = re.compile(r"[ \t]*Art\.?\s*([\dºoA-Z.\-]+)")


def _normalize_art_num(raw: str) -> str:
    s = raw.strip().rstrip(".").replace(".", "")
    s = re.sub(r"[ºo]$", "", s, flags=re.IGNORECASE)
    return s.upper()


def _segment_articles_raw(text: str) -> List[Tuple[str, str]]:
    """(article_number, span_text) pairs from text that still has annotations."""
    matches = list(_ARTIGO.finditer(text))
    out = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        span = text[start:end]
        num_m = _ART_NUM.match(span)
        if num_m:
            out.append((_normalize_art_num(num_m.group(1)), span))
    return out


def extract_article_amendment_years(text: str) -> Dict[str, int]:
    """Article number -> most recent amendment year; never-amended articles are absent."""
    years: Dict[str, int] = {}
    for art_num, span in _segment_articles_raw(text):
        candidates = []
        for m in _AMEND_ANNOTATION.finditer(span):
            candidates.extend(
                int(y) for y in _YEAR_TOKEN.findall(m.group(0)) if 1900 <= int(y) <= 2030)
        if candidates:
            years[art_num] = max(candidates)
    return years


def _strip_boilerplate_lines(text: str) -> str:
    return _BOILERPLATE_LINE.sub("", text)


def _cut_preamble(text: str) -> str:
    """Drop text up to the enacting clause, else up to the first structural anchor."""
    m = _ENACT.search(text[:4000])
    if m:
        return text[m.end():].lstrip(" :\n")
    m = _FIRST_ANCHOR.search(text)
    if m:
        return text[m.start():]
    return text


def _cut_signoff(text: str) -> str:
    """Remove everything from the LAST 'Brasília, <year>' sign-off onward."""
    matches = list(_BRASILIA_SIGNOFF.finditer(text))
    if matches:
        return text[: matches[-1].start()].rstrip()
    return text


# An annex opening, on its own line or as the heading of the block that
# follows the sign-off ("ANEXO", "ANEXO I", "Lista de serviços anexa à …").
_ANNEX_HEADING = re.compile(
    r"(?im)^[ \t]*(?:ANEXO(?:\s+[IVXL\d]+)?(?:\s*[-–].*)?"
    r"|AP[ÊE]NDICE(?:\s+[IVXL\d]+)?"
    r"|LISTA\s+DE\s+SERVI[ÇC]OS.*"
    r"|TABELA(?:\s+[IVXL\d]+)?.*)$"
)
# Signature and publication lines between the sign-off and the annex.
_SIGNATURE_BLOCK = re.compile(
    r"(?im)^[ \t]*(?:[A-ZÀ-Ý][A-ZÀ-Ý .'-]{4,}"
    r"|Este texto n[ãa]o substitui.*|Presidente da Rep[úu]blica.*)$"
)


# A signature line and the names under it; can sit mid-document (CF 1988 before the ADCT).
_SIGNOFF_LINE = re.compile(r"(?im)^[ \t]*Bras[ií]lia\s*,\s*\d{1,2}.*$")
_NAME_LINE = re.compile(r"^[ \t]*[A-ZÀ-Ý][A-ZÀ-Ýa-zà-ÿ .'-]{2,60}$")
# Signatories printed as one long line of "Name - Role," pairs.
_SIGNATORY_RUN = re.compile(r"^[ \t]*,?[^\n]*?(?:\s-\s[^\n]*?){3,}$")


def strip_signature_blocks(text: str) -> str:
    """Remove "Brasília, <date>" signature blocks and the names under them."""

    lines = text.split("\n")
    kept, index = [], 0
    while index < len(lines):
        if _SIGNOFF_LINE.match(lines[index]):
            index += 1
            while index < len(lines) and (
                not lines[index].strip()
                or _NAME_LINE.match(lines[index])
                or _SIGNATORY_RUN.match(lines[index])
            ):
                index += 1
            continue
        kept.append(lines[index])
        index += 1
    return "\n".join(kept)


def split_annexes(text: str) -> Tuple[str, List[str]]:
    """Split an act into (text up to the sign-off, one string per annex after it)."""

    matches = list(_BRASILIA_SIGNOFF.finditer(text))
    if not matches:
        return text, []
    body = strip_signature_blocks(text[: matches[-1].start()]).rstrip()
    tail = text[matches[-1].end():]
    headings = list(_ANNEX_HEADING.finditer(tail))
    if not headings:
        return body, []
    annexes = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(tail)
        block = tail[heading.start(): end].strip()
        if len(block.split()) > 20:
            annexes.append(block)
    return body, annexes


def _strip_annotations(text: str) -> str:
    text = _ANN_REVOKED.sub("(Revogado)", text)
    text = _ANN_REVOKED_OPEN.sub("(Revogado)", text)   # unclosed variant
    # Twice, for chained annotations "(...) (...)".
    text = _ANN_DROP.sub("", text)
    text = _ANN_DROP.sub("", text)
    text = _ANN_DROP_OPEN.sub("", text)                # unclosed variant
    text = _NR_MARKER.sub("", text)
    text = _DOT_LEADERS.sub(" ", text)
    return text


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)          # collapse runs of spaces/tabs
    text = re.sub(r" *\n *", "\n", text)          # trim line edges
    text = re.sub(r"\n{3,}", "\n\n", text)        # cap blank runs
    text = re.sub(r"\(\s*\)", "", text)           # empty parens from stripping
    text = re.sub(r"[ \t]+([.;,:])", r"\1", text)  # space before punctuation
    return text.strip()


def clean_law_text(raw: str) -> Tuple[str, Dict[str, int]]:
    """Clean a law or code; return (operative body, per-article amendment years)."""
    text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_boilerplate_lines(text)
    text = _cut_preamble(text)
    text = _cut_signoff(text)
    amend_years = extract_article_amendment_years(text)
    text = _strip_annotations(text)
    text = _normalize_whitespace(text)
    return text.strip(), amend_years


def clean_constitution_text(raw: str) -> Tuple[str, Dict[str, int]]:
    """Clean the 1988 Constitution, keeping the PREÂMBULO; return (text, amendment years)."""
    text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_boilerplate_lines(text)
    # Keep from the PREÂMBULO / first TÍTULO onward, dropping repeated headers.
    m = re.search(r"(?im)^\s*(PRE[ÂA]MBULO|T[ÍI]TULO\s+I\b)", text)
    if m:
        text = text[m.start():]
    # The signatory roster sits between the body and the ADCT; split so the ADCT is kept.
    adct = _ADCT.search(text)
    if adct:
        body, transitory = text[:adct.start()], text[adct.start():]
        body = _cut_signoff(body)          # strip promulgation + roster
        transitory = _cut_signoff(transitory)
        text = f"{body}\n\n{transitory}"
    else:
        text = _cut_signoff(text)
    amend_years = extract_article_amendment_years(text)
    text = _strip_annotations(text)
    text = _normalize_whitespace(text)
    return text, amend_years


# Treaty "ARTIGO N" captions, rewritten to "Art. N".
_TREATY_ARTIGO = re.compile(r"(?im)^([ \t]*)ARTIGO\b\.?[ \t]*(?=\d)")

# Spelled-out treaty captions ("Artigo Primeiro", "Artigo Dez") -> number.
_PT_ORDINALS = {
    "primeiro": 1, "segundo": 2, "terceiro": 3, "quarto": 4, "quinto": 5,
    "sexto": 6, "setimo": 7, "sétimo": 7, "oitavo": 8, "nono": 9,
    "decimo": 10, "décimo": 10, "vigesimo": 20, "vigésimo": 20,
    "trigesimo": 30, "trigésimo": 30,
}
_PT_CARDINALS = {
    "um": 1, "dois": 2, "tres": 3, "três": 3, "quatro": 4, "cinco": 5,
    "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11,
    "doze": 12, "treze": 13, "catorze": 14, "quatorze": 14, "quinze": 15,
    "dezesseis": 16, "dezessete": 17, "dezoito": 18, "dezenove": 19,
    "vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50,
}
_TREATY_ARTIGO_WORDS = re.compile(
    r"(?im)^([ \t]*)ARTIGO[ \t]+([A-Za-zÀ-ÿ]+(?:[ \t\-]+[A-Za-zÀ-ÿ]+)?)[ \t]*$"
)


def _spelled_article_number(words: str) -> Optional[int]:
    """"Primeiro" -> 1, "Décimo Primeiro" -> 11, "Dez" -> 10; None when not a number."""
    parts = re.split(r"[ \t\-]+", words.strip().lower())
    total = 0
    for part in parts:
        value = _PT_ORDINALS.get(part, _PT_CARDINALS.get(part))
        if value is None:
            return None
        total += value
    return total or None


def _normalize_treaty_captions(text: str) -> str:
    """Rewrite "ARTIGO N" and spelled-out "Artigo Primeiro" captions to "Art. N"."""
    text = _TREATY_ARTIGO.sub(r"\1Art. ", text)

    def repl(m: re.Match) -> str:
        number = _spelled_article_number(m.group(2))
        return f"{m.group(1)}Art. {number}" if number else m.group(0)

    return _TREATY_ARTIGO_WORDS.sub(repl, text)


# Start of the annex / convention body that follows a promulgation sign-off.
_TREATY_ANNEX_ANCHOR = re.compile(
    r"(?im)^[ \t]*(ANN?EXO\b|CONVEN[ÇC][ÃA]O\b|PACTO\b|PROTOCOLO\b|"
    r"LEI\s+UNIFORME\b|PARTE\s+[IVXLCDM]|ARTIGO\b|Art\.)")


def _cut_promulgation_head(text: str) -> str:
    """Drop a promulgation decree's own text up to the first annex anchor after its sign-off."""
    m = _BRASILIA_SIGNOFF.search(text)
    tail = text[m.end():] if m else text
    a = _TREATY_ANNEX_ANCHOR.search(tail)
    if a:
        return tail[a.start():]
    return tail.lstrip(" \n") if m else text


def prepare_treaty_raw(raw: str) -> str:
    """Reduce a treaty-promulgation decree to its annex with "Art. N" captions; annotations kept."""
    text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_boilerplate_lines(text)
    text = _cut_promulgation_head(text)
    return _normalize_treaty_captions(text)


def clean_treaty_text(raw: str) -> Tuple[str, Dict[str, int]]:
    """Clean a treaty-promulgation decree: keep the annex after the sign-off, as "Art. N"."""
    text = prepare_treaty_raw(raw)
    amend_years = extract_article_amendment_years(text)
    text = _strip_annotations(text)
    text = _normalize_whitespace(text)
    return text.strip(), amend_years


# Structure-aware chunking

# Heading matcher: kind + numeral, with the name possibly inline or on the next
# line. Also matches the Constitution's ADCT block as a top-level unit.
_HEADING = re.compile(
    r"(?m)^[ \t]*("
    r"PARTE|LIVRO|T[ÍI]TULO|CAP[ÍI]TULO|SE[ÇC][ÃA]O|SUBSE[ÇC][ÃA]O"
    r")\s+([IVXLCDM]+(?:-[A-Z])?|[ÚU]NICO|GERAL|ESPECIAL|PRELIMINAR)"
    r"\b[ \t]*(.*)$"
)
_ADCT = re.compile(r"(?m)^[ \t]*(ATO DAS DISPOSI[ÇC][ÕO]ES CONSTITUCIONAIS TRANSIT[ÓO]RIAS)")
_ARTIGO = re.compile(r"(?m)^[ \t]*Art\.?\s*\d+")

# An article quoted inside an amendment to another law ("passa a vigorar com a
# seguinte redação: Art. 33.") is not an article boundary of this document.
_QUOTED_AMENDMENT = re.compile(
    r"passa(?:m)?\s+a\s+vigorar\s+com\s+a\s+seguinte\s+reda[çc][ãa]o", re.IGNORECASE)

# Which structural level each heading kind occupies (deeper clears shallower).
_LEVEL = {"PARTE": 0, "LIVRO": 1, "TÍTULO": 2, "TITULO": 2, "CAPÍTULO": 3,
          "CAPITULO": 3, "SEÇÃO": 4, "SECAO": 4, "SEÇAO": 4, "SECÃO": 4,
          "SUBSEÇÃO": 5, "SUBSECAO": 5}
_LEVEL_KEYS = ["parte", "livro", "titulo", "capitulo", "secao", "subsecao"]
_LEVEL_LABEL = {0: "Parte", 1: "Livro", 2: "Título", 3: "Capítulo",
                4: "Seção", 5: "Subseção"}


def _norm_kind(kind: str) -> int:
    key = unicodedata.normalize("NFKD", kind).encode("ascii", "ignore").decode().upper()
    return {"PARTE": 0, "LIVRO": 1, "TITULO": 2, "CAPITULO": 3,
            "SECAO": 4, "SUBSECAO": 5}.get(key, 2)


def _collect_markers(text: str) -> List[dict]:
    """Ordered list of structural markers (headings, ADCT, articles)."""
    markers = []
    for m in _HEADING.finditer(text):
        markers.append({"pos": m.start(), "type": "heading",
                        "level": _norm_kind(m.group(1)),
                        "num": m.group(2).strip(), "inline": m.group(3).strip(),
                        "end": m.end()})
    for m in _ADCT.finditer(text):
        markers.append({"pos": m.start(), "type": "heading", "level": 2,
                        "num": "ADCT", "inline": "Ato das Disposições Constitucionais Transitórias",
                        "end": m.end()})
    for m in _ARTIGO.finditer(text):
        # A quoted amendment stays in the amending article's body.
        lookback = text[max(0, m.start() - 120):m.start()]
        if _QUOTED_AMENDMENT.search(lookback):
            continue
        markers.append({"pos": m.start(), "type": "artigo", "end": m.end()})
    markers.sort(key=lambda x: x["pos"])
    return markers


def _heading_name(text: str, marker: dict) -> str:
    """Heading display name: inline text, or the next non-empty line."""
    if marker["inline"]:
        return marker["inline"]
    tail = text[marker["end"]:marker["end"] + 200].lstrip("\n")
    nxt = tail.split("\n", 1)[0].strip()
    if nxt and not _HEADING.match(nxt) and not _ARTIGO.match(nxt):
        return nxt
    return ""


def _path_string(path: Dict[str, str]) -> str:
    parts = []
    for i, key in enumerate(_LEVEL_KEYS):
        if path.get(key):
            parts.append(f"{_LEVEL_LABEL[i]} {path[key]}")
    return " › ".join(parts)


def parse_articles(text: str, amend_years: Optional[Dict[str, int]] = None
                   ) -> Tuple[str, List[dict]]:
    """Split cleaned text into (lead block, articles with structural path and amendment year)."""
    amend_years = amend_years or {}
    markers = _collect_markers(text)
    art_markers = [m for m in markers if m["type"] == "artigo"]
    if not art_markers:
        return text.strip(), []

    lead = text[: art_markers[0]["pos"]].strip()

    # Walk markers, tracking the running structural path.
    path: Dict[str, str] = {k: "" for k in _LEVEL_KEYS}
    articles: List[dict] = []
    n = len(markers)
    for i, mk in enumerate(markers):
        if mk["type"] == "heading":
            lvl = mk["level"]
            name = _heading_name(text, mk)
            label = f"{mk['num']} – {name}".strip(" –") if name else mk["num"]
            path[_LEVEL_KEYS[lvl]] = label
            for deeper in _LEVEL_KEYS[lvl + 1:]:
                path[deeper] = ""
        else:  # artigo
            start = mk["pos"]
            end = markers[i + 1]["pos"] if i + 1 < n else len(text)
            body = text[start:end].strip()
            if body:
                num_m = _ART_NUM.match(body)
                art_num = _normalize_art_num(num_m.group(1)) if num_m else None
                articles.append({"text": body, "tokens": count_tokens(body),
                                 "path": dict(path),
                                 "last_amended": amend_years.get(art_num)})
    return lead, articles


def _split_oversized(article: dict, trigger: int, size: int) -> List[dict]:
    """Split an article over `trigger` tokens at paragraph, then line, boundaries into ~`size` pieces."""
    if article["tokens"] <= trigger:
        return [article]
    parts = [p.strip() for p in re.split(r"\n{2,}", article["text"]) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in article["text"].split("\n") if p.strip()]

    # Inciso lists often have no blank lines; split oversized parts by line.
    refined: List[str] = []
    for p in parts:
        if count_tokens(p) > trigger:
            lines = [l.strip() for l in p.split("\n") if l.strip()]
            refined.extend(lines if len(lines) > 1 else [p])
        else:
            refined.append(p)
    parts = refined

    out, buf, buf_tok = [], [], 0
    la = article.get("last_amended")
    for p in parts:
        pt = count_tokens(p)
        if buf and buf_tok + pt > size:
            out.append({"text": "\n".join(buf), "tokens": buf_tok, "path": article["path"],
                       "last_amended": la})
            buf, buf_tok = [p], pt
        else:
            buf.append(p)
            buf_tok += pt
    if buf:
        out.append({"text": "\n".join(buf), "tokens": buf_tok, "path": article["path"],
                   "last_amended": la})
    return out or [article]


def _chunk_article_numbers(text: str) -> List[str]:
    """Article numbers whose marker opens a line in this chunk's text."""
    numbers = []
    for m in _ARTIGO.finditer(text):
        num_m = _ART_NUM.match(text[m.start():])
        if num_m:
            num = _normalize_art_num(num_m.group(1))
            if num not in numbers:
                numbers.append(num)
    return numbers


def _same_group(a: Dict[str, str], b: Dict[str, str]) -> bool:
    """Two articles share a chunk-level parent (same título+capítulo+seção)."""
    return all(a.get(k) == b.get(k) for k in ("titulo", "capitulo", "secao"))


def chunk_document(text: str, target: int, hard_max: int, min_tok: int,
                   amend_years: Optional[Dict[str, int]] = None) -> List[dict]:
    """Pack whole articles sharing a structural parent into chunks of about `target` tokens.

    Each chunk carries `last_amended`, the latest amendment year of its articles.
    """
    lead, articles = parse_articles(text, amend_years)

    # No article structure → fall back to paragraph packing on the whole text.
    if not articles:
        pseudo = {"text": text.strip(), "tokens": count_tokens(text),
                  "path": {k: "" for k in _LEVEL_KEYS}, "last_amended": None}
        return _split_oversized(pseudo, trigger=target, size=target)

    # Split oversized articles first.
    units: List[dict] = []
    for art in articles:
        units.extend(_split_oversized(art, trigger=hard_max, size=target))

    chunks: List[dict] = []
    buf: List[dict] = []
    buf_tok = 0

    def flush():
        nonlocal buf, buf_tok
        if buf:
            amend_candidates = [u["last_amended"] for u in buf if u.get("last_amended")]
            chunks.append({
                "text": "\n\n".join(u["text"] for u in buf),
                "tokens": buf_tok,
                "path": buf[0]["path"],
                "n_articles": sum(1 for u in buf if _ARTIGO.match(u["text"])),
                "last_amended": max(amend_candidates) if amend_candidates else None,
            })
            buf, buf_tok = [], 0

    for u in units:
        boundary = buf and not _same_group(buf[-1]["path"], u["path"])
        # Flush on overflow, or at a structural boundary once the chunk is big enough.
        if buf and (buf_tok + u["tokens"] > hard_max
                    or (boundary and buf_tok >= min_tok)
                    or (buf_tok >= target and not boundary and buf_tok + u["tokens"] > target)):
            flush()
        buf.append(u)
        buf_tok += u["tokens"]
    flush()

    # Merge orphan slivers into the previous chunk when they share a parent.
    merged: List[dict] = []
    for ch in chunks:
        if (merged and ch["tokens"] < min_tok
                and _same_group(merged[-1]["path"], ch["path"])
                and merged[-1]["tokens"] + ch["tokens"] <= hard_max):
            prev = merged[-1]
            prev["text"] += "\n\n" + ch["text"]
            prev["tokens"] += ch["tokens"]
            prev["n_articles"] += ch["n_articles"]
            prev["last_amended"] = max(
                [y for y in (prev["last_amended"], ch["last_amended"]) if y], default=None)
        else:
            merged.append(ch)

    # Attach the lead block (before Art. 1) to the first chunk; a large lead is split.
    lead_tokens = count_tokens(lead) if lead else 0
    if lead and lead_tokens > hard_max:
        empty_path = {k: "" for k in _LEVEL_KEYS}
        lead_chunks = _split_oversized(
            {"text": lead, "tokens": lead_tokens, "path": empty_path, "last_amended": None},
            trigger=hard_max, size=target)
        for lc in lead_chunks:
            lc["n_articles"] = 0
        merged = lead_chunks + merged
    elif lead and merged:
        merged[0]["text"] = f"{lead}\n\n{merged[0]['text']}"
        merged[0]["tokens"] = count_tokens(merged[0]["text"])
    elif lead and not merged:
        merged = [{"text": lead, "tokens": lead_tokens,
                   "path": {k: "" for k in _LEVEL_KEYS}, "n_articles": 0,
                   "last_amended": None}]
    return merged


# Pipeline

class OABCorpusPipeline:
    """Select → clean → chunk → export the OAB legal corpus."""

    def __init__(self, data_path: Path = DATA_PATH, csv_filename: str = CSV_FILENAME,
                 export_filename: str = EXPORT_FILENAME,
                 supplementary_dir: str = SUPPLEMENTARY_DIR,
                 include_supplementary: bool = False,
                 target_year: Optional[int] = None):
        self.data_path = Path(data_path)
        self.csv_path = self.data_path / csv_filename
        self.export_path = self.data_path / export_filename
        self.supplementary_dir = self.data_path / supplementary_dir
        self.include_supplementary = include_supplementary
        # Resolve each article to its wording in force at this year; None keeps the newest.
        self.target_year = target_year
        self.version_stats: Dict[str, int] = {}
        self.raw: Optional[pd.DataFrame] = None

    # 1. Load
    def load(self) -> pd.DataFrame:
        logger.info(f"Loading {self.csv_path}")
        self.raw = pd.read_csv(self.csv_path, low_memory=False)
        self.raw["_num"] = self.raw["num_lei"].apply(clean_num)
        logger.info(f"Loaded {len(self.raw):,} raw rows")
        return self.raw

    # 2. Select
    def select(self) -> pd.DataFrame:
        """Pull the 1988 Constitution + the curated registry laws."""
        from src.data.laws_versioning import EntryIntoForce, act_publication_date

        in_force = EntryIntoForce.load(self.data_path / ENTRY_INTO_FORCE)

        logger.info("Selecting OAB corpus (Constitution + curated registry)")
        records: List[dict] = []

        # 1988 Constitution
        const = self.raw[
            (self.raw["tipo"] == "Constituição")
            & self.raw["lei_text"].fillna("").str.contains(
                CONST_1988_MARKER, case=False, regex=False)
        ]
        if const.empty:
            logger.error("1988 Constitution not found!")
        else:
            r = const.iloc[0]
            records.append({
                "num": "1988", "tipo": "Constituição", "label": "Constituição Federal de 1988",
                "area": "Constitucional", "raw_text": str(r["lei_text"]),
                "ementa": "Constituição da República Federativa do Brasil de 1988",
                "main_sancionador": r.get("main_sancionador"), "link": r.get("link"),
                "data_lei": date(1988, 10, 5), "ano": 1988,
                "source": "legislaai", "kind": "constituicao", "slug": "cf_1988",
            })

        # Registry laws
        found, missing = 0, []
        for num, source, tipo, label, area in REGISTRY:
            rows = self.raw[(self.raw["_num"] == num) & (self.raw["tipo"] == source)]
            if rows.empty:
                missing.append((num, label, source))
                continue
            # Fullest text, against short mislabeled variants.
            r = rows.loc[rows["lei_text"].fillna("").str.len().idxmax()]
            parsed = _parse_pt_date(r.get("data_lei"))
            ano = parsed.year if parsed else None
            # Publication date from the act's own text.
            published = act_publication_date(r.get("lei_text"), num, r.get("data_lei"))
            slug = f"{_slug_tipo(tipo)}_{num}_{ano or 'ND'}"
            start = in_force.deferred_start(slug, published)
            valid_from = start.isoformat() if start is not None else None
            records.append({
                "num": num, "tipo": tipo, "label": label, "area": area,
                "raw_text": str(r["lei_text"]),
                "ementa": r.get("ementa") if pd.notna(r.get("ementa")) else label,
                "main_sancionador": r.get("main_sancionador"), "link": r.get("link"),
                "data_lei": published or parsed, "ano": ano,
                "source": "legislaai", "kind": "lei",
                "slug": slug,
                "act_valid_until": REGISTRY_VALID_UNTIL.get((tipo, num)),
                "act_valid_from": valid_from,
            })
            found += 1

        logger.info(f"Selected {found}/{len(REGISTRY)} registry laws + "
                    f"{'1' if not const.empty else '0'} Constitution")
        if missing:
            logger.warning(f"{len(missing)} registry laws not found in raw data:")
            for num, label, source in missing:
                logger.warning(f"  missing: {label} (num={num}, expected in '{source}')")
        logger.info(f"{len(KNOWN_ABSENT)} known-absent high-value statutes "
                    f"(pre-1988 gaps) documented in KNOWN_ABSENT")

        self.selected = pd.DataFrame(records)
        by_area = self.selected.groupby("area").size().sort_values(ascending=False)
        logger.info(f"Corpus by area:\n{by_area}")
        return self.selected

    # 2b. Select supplementary (manually-sourced .txt files)
    def select_supplementary(
        self, target_year=_INHERIT_TARGET_YEAR, apply_validity: bool = True
    ) -> pd.DataFrame:
        """Append the hand-sourced documents in ``supplementary/summary.json`` to ``self.selected``.

        With ``apply_validity`` a document is dropped when not in force at ``target_year``;
        otherwise all are kept and ``valid_until`` is carried as ``act_valid_until``.
        """
        if target_year is _INHERIT_TARGET_YEAR:
            target_year = self.target_year
        summary_path = self.supplementary_dir / SUPPLEMENTARY_SUMMARY
        logger.info(f"Selecting supplementary documents from {summary_path}")
        manifest = json.loads(summary_path.read_text(encoding="utf-8"))
        registry_nums = {num for num, *_ in REGISTRY}

        records, by_kind, replaced = [], {}, []
        for d in manifest["documents"]:
            valid_until = d.get("valid_until")
            valid_from = d.get("valid_from")
            # An unresolved corpus drops repealed acts outright.
            repealed_unresolved = target_year is None and valid_until is not None
            if apply_validity and (repealed_unresolved or not _in_force_in_year(
                valid_from, valid_until, target_year
            )):
                logger.info(f"  skipping {d['id']} — in force "
                            f"{valid_from or '…'}–{valid_until or '…'}, "
                            f"target_year={target_year}")
                continue

            txt_path = self.supplementary_dir / d["path"]
            raw_text = txt_path.read_text(encoding="utf-8")
            num = d.get("num_lei")
            by_kind[d["kind"]] = by_kind.get(d["kind"], 0) + 1
            override = SUPPLEMENTARY_OVERRIDES.get((d["tipo"], num))
            if override:
                replaced.append((d["tipo"], num))
                logger.info(f"  supplementary {d['id']} replaces the scraped "
                            f"text — {override}")
            elif num is not None and num in registry_nums:
                logger.warning(f"  supplementary {d['id']} num_lei={num} also in "
                               f"REGISTRY — intentional coexistence, keeping both")
            records.append({
                "num": num, "tipo": d["tipo"], "label": d["title"], "area": d["area"],
                "raw_text": raw_text,
                "ementa": d.get("note") or d["title"],
                "main_sancionador": None, "link": None,
                "data_lei": _iso_date(d.get("data_lei")),
                "ano": d.get("ano"),
                "source": "manual", "kind": d["kind"], "slug": d["id"],
                "act_valid_until": valid_until,
                "act_valid_from": valid_from,
            })

        if replaced and len(self.selected):
            drop = self.selected.apply(
                lambda row: (row.get("tipo"), row.get("num")) in set(replaced), axis=1
            )
            self.selected = self.selected[~drop].reset_index(drop=True)
            logger.info(f"Dropped {int(drop.sum())} scraped rows replaced by "
                        f"supplementary text")

        supp = pd.DataFrame(records)
        self.selected = pd.concat([self.selected, supp], ignore_index=True)
        logger.info(f"Added {len(supp)} supplementary documents by kind: {by_kind}")
        covered = sorted(
            n for n in KNOWN_ABSENT if n in {d.get("num_lei") for d in manifest["documents"]}
        )
        self.absent_covered = covered
        self.absent_remaining = sorted(set(KNOWN_ABSENT) - set(covered))
        logger.info(f"KNOWN_ABSENT covered by supplementary: {len(covered)}; "
                    f"still absent: {len(self.absent_remaining)}")
        # Lei 4.898/1965 and Lei 13.869/2019 (Abuso de Autoridade) are both kept.
        if 4898 in set(supp["num"].dropna()) and 13869 in registry_nums:
            logger.info("  note: Abuso de Autoridade coexistence — Lei 4.898/1965 "
                        "(manual) + Lei 13.869/2019 (legislaai), both kept")
        return self.selected

    # 3. Clean + 4. Chunk
    def _version_registry(self):
        """Act registry over the selected documents, for quoted amendment blocks."""
        from src.data.laws_citations import ActRegistry, act_number, normalize_tipo

        acts = {}
        for _, doc in self.selected.iterrows():
            num = doc.get("num")
            acts[doc["slug"]] = {
                "tipo": doc["tipo"],
                "tipo_key": normalize_tipo(doc["tipo"]),
                "num": act_number(num),
                "ano": None if pd.isna(doc.get("ano")) else int(doc["ano"]),
                "label": doc["label"],
            }
        return ActRegistry(acts)

    def clean_and_chunk(self) -> pd.DataFrame:
        logger.info("Cleaning and structure-aware chunking")
        out_rows: List[dict] = []

        registry = self._version_registry() if self.target_year is not None else None
        version_totals = {
            "occurrences": 0, "kept": 0, "superseded": 0,
            "not_yet_enacted": 0, "repealed": 0, "quoted": 0,
        }

        for _, doc in self.selected.iterrows():
            kind = doc.get("kind") or "lei"
            slug = doc["slug"]

            act_valid_until = doc.get("act_valid_until")
            if act_valid_until is not None and pd.isna(act_valid_until):
                act_valid_until = None
            act_valid_from = doc.get("act_valid_from")
            if act_valid_from is not None and pd.isna(act_valid_from):
                act_valid_from = None
            if not _in_force_in_year(
                act_valid_from, act_valid_until, self.target_year
            ):
                logger.info(f"Skipping {slug}: in force "
                            f"{act_valid_from or '…'}–{act_valid_until or '…'}, "
                            f"target_year={self.target_year}")
                continue

            # A promulgation decree's content is the treaty annexed after its sign-off.
            if kind == "tratado":
                doc = doc.copy()
                doc["raw_text"] = prepare_treaty_raw(str(doc["raw_text"]))

            # Resolve each article to its wording in force at target_year before cleaning.
            if self.target_year is not None:
                from src.data.laws_versioning import filter_raw_text
                filtered, vstats = filter_raw_text(
                    str(doc["raw_text"]), slug,
                    act_year=None if pd.isna(doc.get("ano")) else int(doc["ano"]),
                    target_year=self.target_year,
                    registry=registry,
                )
                for key in version_totals:
                    version_totals[key] += getattr(vstats, key)
                if vstats.occurrences == 0:
                    raise ValueError(
                        f"{slug}: no article markers parsed from its raw text")
                # No article in force at target_year: the act is out of scope.
                if vstats.kept == 0:
                    logger.debug(f"Skipping {slug}: no articles in force at "
                                 f"{self.target_year}")
                    continue
                doc = doc.copy()
                doc["raw_text"] = filtered

            # Cleaner dispatch by document kind.
            if kind == "constituicao":
                cleaned, amend_years = clean_constitution_text(doc["raw_text"])
            elif kind == "tratado":
                cleaned, amend_years = clean_treaty_text(doc["raw_text"])
            else:  # "lei" and "ato_oab"
                cleaned, amend_years = clean_law_text(doc["raw_text"])

            # Header: CF label only; acts without a number use label + year.
            if kind == "constituicao":
                header = doc["label"]
            elif doc.get("num") is None:
                header = f"{doc['label']} ({doc['ano']})"
            else:
                header = f"{doc['label']} — {display_tipo(doc['tipo'], doc['num'], doc['ano'])}"

            chunks = chunk_document(cleaned, CHUNK_TARGET_TOKENS,
                                    CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS, amend_years)

            for i, ch in enumerate(chunks, 1):
                path = ch["path"]
                path_str = _path_string(path)
                # Breadcrumb header: law + structural location.
                prefix = f"[{header}]"
                if path_str:
                    prefix += f"\n{path_str}"
                chunk_text = re.sub(r"\n{3,}", "\n\n", f"{prefix}\n\n{ch['text']}").strip()

                out_rows.append({
                    "doc_id": f"{slug}_c{i:03d}",
                    "chunk_id": f"c{i:03d}",
                    "chunk_seq": i,
                    "n_chunks": len(chunks),
                    "tipo": doc["tipo"],
                    "lei_label": doc["label"],
                    "area": doc["area"],
                    "num_lei": doc["num"],
                    "ano": doc["ano"],
                    # Cap at target_year: a later year comes from a "Vide" note, not the text.
                    "ano_ultima_alteracao": min(
                        ch.get("last_amended") or doc["ano"],
                        self.target_year or 9999,
                    ),
                    "data_lei_parsed": doc["data_lei"],
                    "main_sancionador": doc["main_sancionador"],
                    "ementa": doc["ementa"],
                    "livro": path.get("livro") or None,
                    "titulo": path.get("titulo") or None,
                    "capitulo": path.get("capitulo") or None,
                    "secao": path.get("secao") or None,
                    "path": path_str or None,
                    "n_articles": ch.get("n_articles", 0),
                    # Article numbers in this chunk, to map chunks to graph nodes.
                    "article_numbers": json.dumps(
                        _chunk_article_numbers(ch["text"]), ensure_ascii=False
                    ),
                    "link": doc["link"],
                    "lei_text_raw": doc["raw_text"] if i == 1 else "",
                    "lei_text_llm": chunk_text,
                    "tokens_llm": count_tokens(chunk_text),
                    "source": doc.get("source", "legislaai"),
                })

        self.version_stats = version_totals
        self.chunks = pd.DataFrame(out_rows)
        # Guard: doc_ids must be unique across the merged corpus.
        dupes = self.chunks["doc_id"][self.chunks["doc_id"].duplicated()].unique()
        if len(dupes):
            raise ValueError(f"Duplicate doc_id(s) in merged corpus: {list(dupes)[:10]}")
        # tokens_raw at document level (whole raw law on its first chunk).
        self.chunks["tokens_raw"] = self.chunks.apply(
            lambda r: count_tokens(r["lei_text_raw"]) if r["chunk_seq"] == 1 else 0, axis=1)
        n_docs = self.chunks["doc_id"].str.replace(r"_c\d+$", "", regex=True).nunique()
        logger.info(f"Produced {len(self.chunks):,} chunks from {n_docs} documents")
        return self.chunks

    # 5. Report + export
    def summarize(self) -> None:
        df = self.chunks
        # Group by doc_id: num_lei is null for CFOAB acts.
        doc_key = df["doc_id"].str.replace(r"_c\d+$", "", regex=True)
        logger.info("── Corpus summary ──")
        logger.info(f"Documents: {doc_key.nunique()} | chunks: {len(df)}")
        if "source" in df.columns:
            by_src_docs = doc_key.groupby(df["source"]).nunique().to_dict()
            by_src_chunks = df.groupby("source").size().to_dict()
            logger.info(f"By source — docs: {by_src_docs} | chunks: {by_src_chunks}")
        logger.info(f"Total LLM tokens: {int(df['tokens_llm'].sum()):,} "
                    f"(raw: {int(df['tokens_raw'].sum()):,})")
        stats = df["tokens_llm"].describe(percentiles=[0.5, 0.9, 0.99])
        logger.info(f"Chunk token stats:\n{stats.round(0)}")
        over = df[df["tokens_llm"] > CHUNK_MAX_TOKENS]
        tiny = df[df["tokens_llm"] < CHUNK_MIN_TOKENS]
        logger.info(f"Chunks > MAX ({CHUNK_MAX_TOKENS}): {len(over)} | "
                    f"< MIN ({CHUNK_MIN_TOKENS}): {len(tiny)}")
        if not over.empty:
            for _, r in over.sort_values("tokens_llm", ascending=False).head(10).iterrows():
                logger.warning(f"  oversized {r['doc_id']}: {r['tokens_llm']} tok "
                               f"({r['lei_label']})")
        chunks_per = df.groupby(doc_key).size()
        top_doc_key = chunks_per.idxmax()
        top_label = df.loc[doc_key == top_doc_key, "lei_label"].iloc[0]
        logger.info(f"Chunks per document — median {chunks_per.median():.0f}, "
                    f"max {chunks_per.max()} ({top_label})")

        amended = df[df["ano_ultima_alteracao"] != df["ano"]]
        logger.info(f"Chunks whose content postdates the document's enactment year "
                    f"(ano_ultima_alteracao != ano): {len(amended):,} ({len(amended)/len(df)*100:.1f}%)")

        if self.version_stats:
            logger.info(
                f"Version resolution (target_year={self.target_year}): "
                + ", ".join(f"{k}={v:,}" for k, v in self.version_stats.items()))

    def export(self) -> None:
        cols = [
            "doc_id", "chunk_id", "chunk_seq", "n_chunks", "tipo", "lei_label",
            "area", "num_lei", "ano", "ano_ultima_alteracao", "data_lei_parsed",
            "main_sancionador", "ementa", "livro", "titulo", "capitulo", "secao",
            "path", "n_articles", "article_numbers", "link", "lei_text_raw",
            "lei_text_llm", "tokens_raw", "tokens_llm", "source",
        ]
        cols = [c for c in cols if c in self.chunks.columns]
        self.chunks[cols].to_csv(self.export_path, index=False)
        logger.success(f"Exported {len(self.chunks):,} chunks -> {self.export_path}")

    def run(self) -> None:
        self.load()
        self.select()
        if self.include_supplementary:
            self.select_supplementary()
        self.clean_and_chunk()
        self.summarize()
        self.export()


def _in_force_in_year(
    valid_from: Union[int, str, None],
    valid_until: Union[int, str, None],
    target_year: Optional[int],
) -> bool:
    """Whether an act was in force on 31 December of the target year (True when no year)."""

    if target_year is None or (valid_from is None and valid_until is None):
        return True
    closing = date(target_year, 12, 31)
    if valid_until is not None and _validity_day(valid_until, first=False) < closing:
        return False
    if valid_from is not None and _validity_day(valid_from, first=True) > closing:
        return False
    return True


def _validity_day(value: Union[int, str], first: bool) -> date:
    """Resolve a bound (``2023`` or ``"2016-08-31"``) to a day; ``first`` takes a bare year's first day."""

    text = str(value)
    if "-" in text:
        return date.fromisoformat(text)
    year = int(text[:4])
    return date(year, 1, 1) if first else date(year, 12, 31)


def _iso_date(value: Optional[str]) -> Optional[date]:
    """Parse an ISO date string, passing through None."""

    return date.fromisoformat(value) if value else None


def _slug_tipo(tipo: str) -> str:
    return {"Lei Ordinária": "lei", "Lei Complementar": "lc",
            "Decreto-Lei": "dl", "Constituição": "cf",
            "Emenda Constitucional": "ec", "Decreto": "decreto"}.get(tipo, "doc")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Build the OAB laws corpus (LegislaAI +/- supplementary).")
    parser.add_argument(
        "--with-supplementary", action="store_true",
        help="Also ingest the manually-sourced docs in supplementary/ (-> laws_v2).")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV filename under data/laws/ "
             "(default: laws_v1.csv, or laws_v2.csv with --with-supplementary).")
    parser.add_argument(
        "--supplementary-dir", default=SUPPLEMENTARY_DIR,
        help="Supplementary directory under data/laws/ (default: supplementary).")
    parser.add_argument(
        "--target-year", type=int, default=None,
        help="Resolve each article to the version in force at this year "
             "(-> laws_v3_<year>.csv). Omit for the v1/v2 behaviour.")
    args = parser.parse_args()

    if args.target_year:
        default_export = EXPORT_TEMPLATE_RESOLVED.format(
            version=RESOLVED_VERSION, target_year=args.target_year)
    else:
        default_export = (
            EXPORT_FILENAME_V2 if args.with_supplementary else EXPORT_FILENAME)

    OABCorpusPipeline(
        export_filename=args.output or default_export,
        supplementary_dir=args.supplementary_dir,
        include_supplementary=args.with_supplementary,
        target_year=args.target_year,
    ).run()


if __name__ == "__main__":
    main()
