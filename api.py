from flask import Flask, render_template, jsonify, request
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import os

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

        # Convertir a HTML
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True  # Ocultar el código y mostrar solo resultados
        body, resources = html_exporter.from_notebook_node(notebook_content)
        return body
    except Exception as e:
        print(f"Error al ejecutar/convertir el notebook {notebook_path}: {str(e)}")
        return None

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
