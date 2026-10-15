# tomathec
projeto de modulo da industria

TomaTech | Classificador de Qualidade de Tomates

TomaTech é um sistema inteligente de triagem e classificação de tomates desenvolvido com Python (Flask) para o backend, processamento de imagem com OpenCV e uma interface web moderna.

Toda a análise visual considera somente a região do tomate: o fundo da imagem é removido automaticamente por segmentação (cor e GrabCut) antes de qualquer cálculo de cor, tamanho, textura ou detecção de defeitos.

🛠️ Requisitos de Instalação e Configuração

Para executar o projeto com sucesso em sua máquina, siga atentamente o passo a passo detalhado abaixo.

Passo 1: Baixar e Instalar o Python 3.12

O TomaTech foi desenvolvido para rodar na versão mais recente e estável do Python. Caso não tenha instalado:

Acesse o site oficial de downloads: python.org/downloads

Baixe o instalador oficial do Python 3.12 (ou versão superior compatível).

Importante durante a instalação: No início do assistente de instalação, certifique-se de marcar a caixa "Add python.exe to PATH" (Adicionar o Python ao PATH do sistema). Isso é essencial para conseguir rodar os comandos no terminal.

Para confirmar se a instalação deu certo, abra o seu terminal (Prompt de Comando, PowerShell ou Terminal do VS Code) e digite:

python --version


O retorno deve indicar a versão do Python 3.12 instalada.

Passo 2: Clonar ou Baixar o Repositório

Baixe todos os arquivos do projeto (incluindo app.py, requirements.txt, index.html, etc.) e coloque-os dentro de uma pasta dedicada no seu computador.

Abra o seu terminal e navegue até a pasta onde salvou os arquivos do projeto:

cd caminho/para/a/pasta/do/projeto


Passo 3: Criar e Ativar um Ambiente Virtual (Opcional, mas Recomendado)

Recomenda-se criar um ambiente virtual isolado para instalar as dependências do projeto sem conflitos com outras aplicações Python:

No Windows:

python -m venv venv
venv\Scripts\activate


No macOS / Linux:

python3 -m venv venv
source venv/bin/activate


Passo 4: Instalar as Dependências

Com o ambiente virtual ativado, instale todas as bibliotecas necessárias listadas no arquivo requirements.txt (incluindo Flask, OpenCV, NumPy e PySerial) executando o seguinte comando:

pip install -r requirements.txt


Passo 5: Executar a Aplicação

Após a conclusão da instalação das dependências, inicie o servidor Flask localmente:

python app.py


O terminal exibirá uma mensagem indicando que o servidor está rodando:

Dataset: .../dataset
Abra http://127.0.0.1:5000


Abra o seu navegador de preferência e acesse: http://127.0.0.1:5000 para interagir com o painel de triagem e classificação de tomates.