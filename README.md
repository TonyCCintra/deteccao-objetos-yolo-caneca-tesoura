# deteccao-objetos-yolo-caneca-tesoura
Implementação de um detector de objetos YOLOv8 treinado para identificar "Canecas" e "Tesouras". O projeto cobre a rotulagem de dados com Label Studio, treinamento em Google Colab, e demonstração dos resultados da detecção.

# Detecção de Objetos (Caneca e Tesoura) com YOLOv8

## Descrição do Projeto

Este projeto demonstra o processo completo de treinamento de um modelo de detecção de objetos YOLOv8 para identificar duas classes personalizadas: "Caneca" e "Tesoura". O trabalho envolveu a rotulagem de um dataset de 60 imagens utilizando o Label Studio, a configuração do ambiente de treinamento no Google Colab com GPU (Tesla T4), e a aplicação de transfer learning a partir de um modelo YOLOv8n pré-treinado. O objetivo principal foi cumprir os requisitos de um desafio de projeto da DIO, incluindo a obtenção de um modelo funcional capaz de detectar as novas classes em imagens.

## Tecnologias Utilizadas

*   Python 3
*   YOLOv8 (Ultralytics)
*   Label Studio (para rotulagem de imagens)
*   Google Colab (para treinamento e inferência com GPU)
*   Google Drive (para armazenamento de datasets e resultados)
*   OpenCV (usado implicitamente pelo YOLO)
*   PyTorch (usado implicitamente pelo YOLO)

## Estrutura do Repositório

*   `notebooks/`: Contém o Jupyter Notebook (`Dio_Treinamento_YOLOv8_Caneca_Tesoura.ipynb`) com todo o código para treinamento e inferência.
*   `dataset_config/`:
    *   `data.yaml`: Arquivo de configuração do dataset para o YOLO.
    *   `classes.txt`: Lista das classes ("Caneca", "Tesoura").
*   `sample_results/`:
    *   `images_with_detections/`: Exemplos de imagens de teste com as detecções realizadas pelo modelo treinado.
    *   `training_plots/`: Gráficos gerados durante o treinamento (ex: `results.png`, `confusion_matrix.png`).
*   `model_weights/`: (Opcional) Contém o arquivo `best.pt` com os pesos do melhor modelo treinado. [SE FOR GRANDE, SUBSTITUA POR: Os pesos do modelo (`best.pt`) podem ser encontrados [AQUI](LINK_PARA_SEU_GOOGLE_DRIVE_COM_O_BEST.PT)].
*   `README.md`: Este arquivo.

## Como Rodar o Projeto (Treinamento e Inferência)

1.  **Preparar o Dataset:**
    *   As imagens foram rotuladas usando o Label Studio e exportadas no formato YOLO.
    *   O dataset consiste em 60 imagens, divididas em 48 para treino e 12 para validação.
    *   Os arquivos de imagem e rótulo foram organizados no Google Drive conforme a estrutura esperada pelo `data.yaml`.
2.  **Ambiente:**
    *   O notebook `notebooks/Dio_Treinamento_YOLOv8_Caneca_Tesoura.ipynb` foi projetado para rodar no Google Colab com um ambiente de GPU (Tesla T4 recomendado).
3.  **Executar o Notebook:**
    *   Abra o notebook no Google Colab.
    *   Certifique-se de que o acelerador de hardware está configurado para GPU.
    *   Execute as células em ordem para:
        *   Montar o Google Drive.
        *   Instalar a biblioteca `ultralytics`.
        *   Realizar o treinamento do modelo (os caminhos para `data.yaml` e para salvar os resultados estão configurados para uma estrutura específica no Google Drive).
        *   Realizar a inferência em imagens de teste.

## Resultados do Treinamento

O modelo foi treinado por 50 épocas utilizando o YOLOv8n como base. As principais métricas de validação para o arquivo `best.pt` foram:

| Classe   | Imagens | Instâncias | Precision (P) | Recall (R) | mAP@.5 | mAP@.5:.95 |
| :------- | :------ | :--------- | :------------ | :--------- | :----- | :--------- |
| all      | 12      | 12         | 0.582         | 0.405      | 0.41   | 0.158      |
| Caneca   | 6       | 6          | 0.736         | 0.667      | 0.73   | 0.291      |
| Tesoura  | 6       | 6          | 0.428         | 0.143      | 0.0909 | 0.0259     |

*(Insira aqui o gráfico `results.png` ou outros gráficos relevantes)*
`![Resultados do Treinamento](sample_results/training_plots/results.png)`
`![Matriz de Confusão](sample_results/training_plots/confusion_matrix.png)`

**Análise:** O modelo demonstrou um bom aprendizado para a classe "Caneca", alcançando um mAP@0.5 de 0.73. O desempenho para a classe "Tesoura" foi inferior (mAP@0.5 de 0.09), indicando que o modelo teve mais dificuldade com esta classe, possivelmente devido à menor quantidade de características distintivas visuais, variações de forma, ou oclusões nas imagens de treino/validação para este objeto em um dataset pequeno.

## Demonstração Visual (Exemplos de Detecção)

Abaixo estão exemplos de detecções realizadas pelo modelo treinado em imagens de teste:

*(Insira aqui uma ou duas das suas melhores imagens com detecção)*
`![Detecção Exemplo 1](sample_results/images_with_detections/minha_imagem_teste2.jpeg)` 
*(Legenda: Detecção de "Caneca" em uma imagem de teste.)*

## Desafios Enfrentados e Aprendizados

Durante o desenvolvimento deste projeto, diversos desafios foram encontrados, principalmente relacionados à correta configuração dos dados e do ambiente:
*   **Correspondência de Nomes de Arquivo:** Garantir que os nomes dos arquivos de imagem e seus respectivos arquivos de rótulo `.txt` fossem idênticos (ignorando a extensão) foi crucial e exigiu um processo de "começar do zero" na organização dos dados.
*   **Configuração do `data.yaml`:** Assegurar que os caminhos para os datasets de treino/validação e os nomes das classes estivessem corretos.
*   **Ambiente Google Colab:** Lidar com a montagem do Google Drive, seleção de GPU e reinícios de ambiente que exigiam reinstalação de bibliotecas.
*   **Depuração do "No Labels Found":** Um processo iterativo de verificação de caminhos, nomes de arquivos, conteúdo dos rótulos e arquivos de cache do YOLO.

Esses desafios reforçaram a importância da atenção meticulosa aos detalhes na preparação de dados para projetos de Machine Learning.

## Autor

Tony Cajaiba Cintra
*   GitHub: https://github.com/TonyCCintra/
