from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from eraser import process_pdf, process_image, visualize_red_regions, convert_pdf_to_images
import uuid
import shutil
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
import numpy as np
import cv2
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['SESSION_EXPIRY'] = timedelta(hours=24)  # Sessions expire after 24 hours

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Store session creation times
session_times = {}

def cleanup_old_sessions():
    """Remove sessions older than the expiry time."""
    while True:
        try:
            current_time = datetime.now()
            for session_id in list(session_times.keys()):
                if current_time - session_times[session_id] > app.config['SESSION_EXPIRY']:
                    # Remove session directories
                    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
                    processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
                    
                    if os.path.exists(session_dir):
                        shutil.rmtree(session_dir)
                    if os.path.exists(processed_dir):
                        shutil.rmtree(processed_dir)
                    
                    # Remove from session tracking
                    del session_times[session_id]
            
            # Sleep for 1 hour before next cleanup
            time.sleep(3600)
        except Exception as e:
            print(f"Error in cleanup: {e}")
            time.sleep(300)  # Sleep for 5 minutes if there's an error

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
cleanup_thread.start()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    # Get batch name from form data
    batch_name = request.form.get('batch_name', '')
    
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Record session creation time
    session_times[session_id] = datetime.now()
    
    processed_files = []
    total_files = len(files)
    processed_count = 0
    
    for file_index, file in enumerate(files, 1):
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(session_dir, filename)
            file.save(filepath)
            
            # Process the file
            if filename.lower().endswith('.pdf'):
                # Convert PDF to images
                images = convert_pdf_to_images(filepath)
                total_pages = len(images)
                
                for page_num, image in enumerate(images, 1):
                    # Process each page
                    visualization, mask = visualize_red_regions(image)
                    processed = process_image(image, mask)
                    
                    # Save original and processed images
                    page_filename = f"page_{page_num}.png"
                    original_path = os.path.join(processed_dir, 'original_' + page_filename)
                    processed_path = os.path.join(processed_dir, 'processed_' + page_filename)
                    image.save(original_path)
                    processed.save(processed_path)
                    
                    processed_files.append({
                        'original': 'original_' + page_filename,
                        'processed': 'processed_' + page_filename,
                        'name': f"{filename} - Page {page_num}"
                    })
                    processed_count += 1
            else:
                # Process single image
                img = Image.open(filepath)
                # Preserve EXIF orientation
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif = dict(img._getexif().items())
                    if 274 in exif:  # 274 is the orientation tag
                        if exif[274] == 3:
                            img = img.rotate(180, expand=True)
                        elif exif[274] == 6:
                            img = img.rotate(270, expand=True)
                        elif exif[274] == 8:
                            img = img.rotate(90, expand=True)
                
                visualization, mask = visualize_red_regions(img)
                processed = process_image(img, mask)
                
                # Save original and processed images
                original_path = os.path.join(processed_dir, 'original_' + filename)
                processed_path = os.path.join(processed_dir, 'processed_' + filename)
                img.save(original_path)
                processed.save(processed_path)
                
                processed_files.append({
                    'original': 'original_' + filename,
                    'processed': 'processed_' + filename,
                    'name': filename
                })
                processed_count += 1
    
    return jsonify({
        'session_id': session_id,
        'files': processed_files,
        'total_files': total_files,
        'processed_count': processed_count
    })

@app.route('/download_pdf/<session_id>')
def download_pdf(session_id):
    processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
    if not os.path.exists(processed_dir):
        return jsonify({'error': 'Session not found'}), 404
    
    # Get batch name from query parameter
    batch_name = request.args.get('name', 'processed_images')
    
    # Create PDF
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    
    # Get all processed images
    processed_images = []
    for filename in os.listdir(processed_dir):
        if filename.startswith('processed_'):
            processed_images.append(os.path.join(processed_dir, filename))
    
    # Sort images by name
    processed_images.sort()
    
    # Add each image to PDF
    for img_path in processed_images:
        img = Image.open(img_path)
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Get page dimensions
        page_width, page_height = c._pagesize
        
        # Calculate scaling to fit page while maintaining aspect ratio
        width_ratio = page_width / img_width
        height_ratio = page_height / img_height
        scale = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = img_width * scale
        new_height = img_height * scale
        
        # Center image on page
        x = (page_width - new_width) / 2
        y = (page_height - new_height) / 2
        
        # Add image to PDF
        c.drawImage(img_path, x, y, width=new_width, height=new_height)
        c.showPage()
    
    c.save()
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{batch_name}.pdf",
        mimetype='application/pdf'
    )

@app.route('/session/<session_id>')
def get_session_files(session_id):
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
    
    if not os.path.exists(session_dir):
        return jsonify({'error': 'Session not found'}), 404
    
    files = []
    for filename in os.listdir(processed_dir):
        if filename.startswith('original_') or filename.startswith('processed_'):
            files.append({
                'name': filename,
                'url': f'/file/{session_id}/{filename}'
            })
    
    return jsonify({'files': files})

@app.route('/file/<session_id>/<filename>')
def get_file(session_id, filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], session_id, filename))

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    processed_dir = os.path.join(app.config['PROCESSED_FOLDER'], session_id)
    
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    
    return jsonify({'status': 'success'})

@app.route('/rotate_image/<session_id>/<filename>', methods=['POST'])
def rotate_image(session_id, filename):
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], session_id, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    img = Image.open(filepath)
    # Rotate image 90 degrees clockwise
    rotated = img.rotate(-90, expand=True)
    
    # Save back to the same file
    rotated.save(filepath)
    
    return jsonify({
        'status': 'success',
        'filename': filename
    })

@app.route('/fix_spots/<session_id>/<filename>', methods=['POST'])
def fix_spots(session_id, filename):
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], session_id, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    img = Image.open(filepath)
    # Apply additional processing for spot fixing
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Apply bilateral filter to remove spots while preserving edges
    processed = cv2.bilateralFilter(img_array, 9, 75, 75)
    
    # Convert back to PIL Image
    processed_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    processed_img.save(filepath)
    
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True) 