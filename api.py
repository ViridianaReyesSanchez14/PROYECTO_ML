from flask import Flask, render_template, jsonify
import nbformat
from nbconvert import HTMLExporter

app = Flask(__name__)

# Lista de notebooks, incluyendo uno nuevo
notebooks = [
    "3501_RegresionLineal.ipynb",
    "3501_Regresion_Logistica.ipynb",
    "3501_Visualizacion-de-Datos.ipynb",
    "3501_Preparacion-del-DataSet.ipynb",
    "3501_Creacion-de-Transformadores-y-Pipelines-Personalizados.ipynb",
    "3501_Evaluacion-de-Resultados.ipynb",
    "3501_Support-Vector-Machine.ipynb",
    "Arboles_de_decision.ipynb"  # Nuevo notebook agregado
]

# Convertir notebook a HTML
def convert_notebook_to_html(notebook_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)

        html_exporter = HTMLExporter()
        body, resources = html_exporter.from_notebook_node(notebook_content)
        return body
    except Exception as e:
        print(f"Error al convertir el notebook {notebook_path}: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html', notebooks=notebooks)

@app.route('/notebook/<notebook_name>')
def view_notebook(notebook_name):
    notebook_path = f"notebooks/{notebook_name}"
    try:
        notebook_html = convert_notebook_to_html(notebook_path)
        return render_template('notebook_viewer.html', notebook_html=notebook_html)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)

