from flask import Flask, render_template, jsonify, request
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import os
import base64

app = Flask(__name__)

# Directorio de notebooks
NOTEBOOK_DIR = "notebooks"

# Lista dinámica de notebooks
def get_notebooks():
    return [f for f in os.listdir(NOTEBOOK_DIR) if f.endswith(".ipynb")]

# Ejecutar y convertir notebook a HTML
def execute_and_convert_notebook(notebook_path):
    try:
        # Leer el notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)

        # Ejecutar el notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(notebook_content, {'metadata': {'path': NOTEBOOK_DIR}})

        # Guardar los resultados ejecutados en el archivo
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(notebook_content, f)

        # Filtrar el contenido para incluir solo gráficos, imágenes y métricas clave
        filtered_content = []
        for cell in notebook_content['cells']:
            if cell['cell_type'] == 'code' and 'outputs' in cell:
                for output in cell['outputs']:
                    # Capturar gráficos en formato PNG
                    if output['output_type'] in ['display_data', 'execute_result'] and 'image/png' in output['data']:
                        img_base64 = base64.b64encode(output['data']['image/png']).decode('utf-8')
                        filtered_content.append(f"<img src='data:image/png;base64,{img_base64}' alt='Gráfico'>")
                    # Capturar métricas clave
                    elif output['output_type'] == 'stream' and 'text' in output:
                        metrics = ['f1_score', 'accuracy', 'precision', 'recall', 'roc_auc']
                        if any(metric in output['text'].lower() for metric in metrics):
                            filtered_content.append(f"<p>{output['text']}</p>")

        # Generar HTML con el contenido filtrado
        html_content = """<html><body><h1>Resultados del Notebook</h1>"""
        html_content += "".join(filtered_content)
        html_content += "</body></html>"

        return html_content
    except Exception as e:
        print(f"Error al ejecutar/convertir el notebook {notebook_path}: {str(e)}")
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"

@app.route('/')
def index():
    notebooks = get_notebooks()
    return render_template('index.html', notebooks=notebooks)

@app.route('/notebook/<notebook_name>')
def view_notebook(notebook_name):
    notebook_path = os.path.join(NOTEBOOK_DIR, notebook_name)
    try:
        notebook_html = execute_and_convert_notebook(notebook_path)
        return render_template('notebook_viewer.html', notebook_html=notebook_html)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_notebook', methods=['POST'])
def add_notebook():
    notebook_name = request.json.get("notebook_name")
    if not notebook_name.endswith(".ipynb"):
        notebook_name += ".ipynb"
    notebook_path = os.path.join(NOTEBOOK_DIR, notebook_name)

    # Crear un notebook vacío si no existe
    if not os.path.exists(notebook_path):
        nb = nbformat.v4.new_notebook()
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    return jsonify({"message": "Notebook creado", "notebook": notebook_name})

if __name__ == '__main__':
    app.run(debug=True, port=8000)

