
## 🇷🇺 Запуск (Windows / Linux)

```bash
git clone https://github.com/daniilzaj/banking-rec-system.git
cd banking-rec-system
cd data-ai-service
```
Скопировать входные файлы в banking-rec-system/data-ai-service/data_input

## Запуск Pyton

```bash
cd data-ai-service
pip install -r requirements.txt
python -m app.main
```
# Результаты будут в data_output:
# - recommendations_final.csv
# - recommendations_full_analysis.csv


## Запуск (Docker)

```bash
git clone https://github.com/daniilzaj/banking-rec-system.git
cd banking-rec-system
cd data-ai-service
```
Скопировать входные файлы в banking-rec-system/data-ai-service/data_input
```bash
cd data-ai-service
docker-compose up --build
```
# Результаты будут в data_output:
# - recommendations_final.csv
# - recommendations_full_analysis.csv


