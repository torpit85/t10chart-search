# T-10 Chart Search Engine

## Quick start
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild/refresh the database from your ZIP
```bash
python build_db.py --zip "T-10 Chart reformatted clean.zip" --db t10.sqlite --schema schema.sql --rebuild
```
