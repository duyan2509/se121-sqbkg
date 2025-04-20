# se121-sqbkg
Semantic Query base on Knowledge Graph 
## Install package
```
pip install -r requirements.txt
```
## Set up VnCoreNLP
```
python PrepareVnCoreNLP.py
```
## Enviroment Variables
Create .env with below content
```
VNCORENLP_DIR="get from console after run PrepareVnCoreNLP.py"
GOOGLE_API_KEY=google_api_key
NEO4J_URI=neo4j_uri
NEO4J_USERNAME=neo4j_username
NEO4J_PASSWORD=neo4j_password
```
