import os
from flask import Flask, request, jsonify, render_template
from rag_engine import (
    extract_text_from_pdf,
    split_into_chunks,
    create_vector_store,
    answer_question
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files allowed'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the PDF
    text = extract_text_from_pdf(filepath)
    chunks = split_into_chunks(text)
    create_vector_store(chunks)

    return jsonify({
        'message': f'PDF processed successfully! '
                   f'{len(chunks)} chunks created.',
        'chunks': len(chunks)
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    answer, source_docs = answer_question(question)

    # Extract source page content snippets
    sources = [doc.page_content[:200] + "..." for doc in source_docs]

    return jsonify({
        'answer': answer,
        'sources': sources
    })

if __name__ == '__main__':
    app.run(debug=True)