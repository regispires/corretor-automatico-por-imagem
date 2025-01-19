import glob
import cv2
import numpy as np
import utils
import config as cfg
import pandas as pd

df = pd.read_csv('alunos.csv', sep='\t', dtype={'Matrícula': str})
alunos = dict(zip(df.loc[:, 'Matrícula'], df.loc[:, 'Nome']))

print("Gabarito: ", cfg.gabarito)

def corrigir(nome_do_arquivo, ordem_matr):
    img = cv2.imread(nome_do_arquivo)
    
    # Redimensiona a imagem para um tamanho padrão 3:4
    print("Redimensiona a imagem para um tamanho padrão 3:4")
    img = cv2.resize(img, (1512, 2016))
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_copy = img.copy()

    # Pre Processamentos
    print("Pre Processamentos")
    imagem_sem_sombra = utils.remover_sombra(img)
    imagem_cinza = cv2.cvtColor(imagem_sem_sombra, cv2.COLOR_BGR2GRAY)
    imagem_com_desfoque = cv2.GaussianBlur(imagem_cinza, (3, 3), 5)
    imagem_binaria = cv2.threshold(
        imagem_com_desfoque, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Encontra os contornos das questões e o retangulo maior
    print("Encontra os contornos das questões e o retangulo maior")
    contornos, _ = cv2.findContours(
        imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    retangulos = utils.encontrar_retangulos(contornos)
    maior_retangulo = retangulos[0]
    vertices_maior_retangulo = utils.encontrar_vertices(maior_retangulo)
    vertices_ordenadas = utils.reordenar_pontos(vertices_maior_retangulo)

    cv2.drawContours(img_copy, vertices_ordenadas, -1, (255, 0, 0), 60)
    #cv2.imshow("img_copy", img_copy)

    # Corrige perspectiva da Imagem
    print("Corrige perspectiva da Imagem")
    vertices_float_32 = np.float32(vertices_ordenadas)
    # pinta a borda de preto para remover-la da área de interesse
    print("pinta a borda de preto para remover-la da área de interesse")
    cv2.drawContours(imagem_binaria, [maior_retangulo], -1, (0, 0, 0), 16)

    template_formato_retangulo = np.float32(
        [[0, 0], [267*cfg.NUMERO_ALTERNATIVAS, 0], [0, 193*cfg.NUMERO_QUESTOES], [267*cfg.NUMERO_ALTERNATIVAS, 193*cfg.NUMERO_QUESTOES]])  # shape conhecido do retangulo em pixels
    matriz_de_transformacao = cv2.getPerspectiveTransform(
        vertices_float_32, template_formato_retangulo)
    img_corrigida = cv2.warpPerspective(
        imagem_binaria, matriz_de_transformacao, (267*cfg.NUMERO_ALTERNATIVAS, 193*cfg.NUMERO_QUESTOES))
    img_bordas_cortadas = utils.cortar_imagem(
        img_corrigida, 0.99)  # corta 1% das bordas e mantém 99% da imagem

    img_linhas = utils.fatiar_vertical(img_bordas_cortadas, cfg.NUMERO_QUESTOES)

    respostas = []
    anuladas = []
    pontuacao = 0
    print("Corrigindo questões")
    for indice_linha, linha in enumerate(img_linhas):
        print("Corrigindo questão", indice_linha + 1)
        indice_marcado = -1
        img_colunas = utils.fatiar_horizontal(linha, cfg.NUMERO_ALTERNATIVAS)
        numero_pixels_na_coluna = []
        for indice_coluna, coluna in enumerate(img_colunas):
            coluna = utils.cortar_imagem(coluna, 0.90)
            numero_de_pixels_brancos = cv2.countNonZero(coluna)
            numero_pixels_na_coluna.append(numero_de_pixels_brancos)
            if (cfg.DEBUGAR):
                cv2.imshow("imagem_circulo"+str(indice_linha) +
                           "_" + str(indice_coluna)+"_"+str(numero_de_pixels_brancos), coluna)
        numero_pixels_na_coluna_sem_maior = numero_pixels_na_coluna.copy()
        numero_pixels_na_coluna_sem_maior.remove(max(numero_pixels_na_coluna))
        print('numero de pixels nas alternativas a,b,c,d,e:', numero_pixels_na_coluna[::-1])
        alternativas_marcadas_na_questao = 0
        metade_max_de_pixels_alternativas = max(numero_pixels_na_coluna)/2
        for indice_pixels, pixels in enumerate(numero_pixels_na_coluna):
            if (pixels > metade_max_de_pixels_alternativas):
                alternativas_marcadas_na_questao += 1
                indice_marcado = cfg.NUMERO_ALTERNATIVAS - 1 - indice_pixels
                alternativa_marcada = utils.obter_alternativa_pelo_indice(indice_marcado)
                print('alternativa_marcada:', alternativa_marcada)
            
        if (alternativas_marcadas_na_questao > 1):
            print('>> Mais de uma alternativa marcada. Questão anulada.')
            indice_marcado = -1
            alternativa_marcada = 'X'
            anuladas.append({'ordem_matr': ordem_matr, 'questao': indice_linha + 1, 
                        'numero_pixels_na_coluna': numero_pixels_na_coluna[::-1]})
        #else:
        #    alternativa_marcada = utils.obter_alternativa_pelo_indice(indice_marcado)
        respostas.append(alternativa_marcada)
        print('>> Resposta marcada:', alternativa_marcada)
        if (len(cfg.gabarito) == cfg.NUMERO_QUESTOES and indice_marcado == utils.obter_indice_da_alternativa(cfg.gabarito[indice_linha])):
            pontuacao += 1

    if cfg.DEBUGAR:
        # Imagem binária deve ter o retângulo das questões bem destacado e as opções marcadas também
        # Nos contornos o retângulo das questões deve ser o maior retângulo destacado da imagem
        copia_contornos = img.copy()
        cv2.drawContours(copia_contornos, contornos, -1, (0, 0, 255), 3)
        cv2.drawContours(copia_contornos, [
                         maior_retangulo], -1, (0, 255, 0), 20)
        utils.exibir_imagens([('img', img), ('imagem_sem_sombra', imagem_sem_sombra), (
            'imagem_com_desfoque', imagem_com_desfoque), ('imagem_binaria', imagem_binaria), ('contornos', copia_contornos), ('img_corrigida', img_corrigida), ('img_bordas_cortadas', img_bordas_cortadas)])
        print("DEBUG Respostas: ", "Arquivo: ", nome_do_arquivo,
              respostas, "nota: ", pontuacao, "/ 10")
        print("FIM Debug \n\n\n")
        cv2.waitKey(0)

    return respostas, pontuacao, anuladas

arquivos_a_serem_corrigidos = glob.glob(cfg.input_dir + '/*.jpg')

# Definir o caminho do arquivo CSV para salvar os dados
caminho_csv = "resultados.csv"

# Criar e abrir o arquivo CSV no modo de escrita
with open(caminho_csv, mode='w', encoding='utf-8') as arquivo_csv:
    # Escrever o cabeçalho manualmente
    questoes = list(range(1, cfg.NUMERO_QUESTOES + 1))
    questoes = ['Q' + str(n_questao) for n_questao in questoes]
    questoes = ','.join(map(str, questoes))
    questoes = questoes.replace(',', ';')
    arquivo_csv.write(f"Ordem;Matricula;Nome;{questoes};Nota;Pontos;N_Questoes\n")

    # Processar cada arquivo na lista
    questoes_anuladas = []
    for arquivo in arquivos_a_serem_corrigidos:
        print('Corrigir:', arquivo)
        ordem_matr = arquivo.replace(
            cfg.input_dir + '/', "").replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
        print(ordem_matr)
        respostas, pontuacao, anuladas = corrigir(arquivo, ordem_matr)
        if (len(anuladas) > 0):
            questoes_anuladas.append(anuladas)
        ordem, matr = ordem_matr.split('-')
        ordem = int(ordem)
        nota = pontuacao / cfg.NUMERO_QUESTOES * 10

        # Escrever os dados no arquivo CSV manualmente
        linha = f"{ordem},{matr},{alunos[matr]},{','.join(respostas)},{round(nota, 2)},{pontuacao},{cfg.NUMERO_QUESTOES}\n"
        linha = linha.replace(',', ';').replace('.', ',')
        arquivo_csv.write(linha)

print(f"Os resultados foram salvos no arquivo '{caminho_csv}'.")
df_result = pd.read_csv(caminho_csv, sep=';', usecols=['Matricula', 'Nome'], dtype={'Matricula': str})
faltantes = df[~df['Matrícula'].isin(df_result['Matricula'])]
print('Alunos faltantes:')
print(faltantes)
print('Questões anuladas:', questoes_anuladas)