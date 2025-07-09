#!/usr/bin/env python3
"""
Simple HTTP server for AI Image-to-Video Generator
This is a minimal server that works without heavy dependencies
"""

import http.server
import socketserver
import json
import os
import sys
from pathlib import Path

PORT = 8000

class AIVideoHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Image-to-Video Generator</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
                    .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
                    .success { background: #d4edda; color: #155724; }
                    .warning { background: #fff3cd; color: #856404; }
                    .info { background: #d1ecf1; color: #0c5460; }
                    .error { background: #f8d7da; color: #721c24; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🎬 AI Image-to-Video Generator</h1>
                    <p>Simple Server Mode</p>
                </div>
                
                <div class="section">
                    <h2>📊 System Status</h2>
                    <div class="status info">
                        <strong>Server:</strong> Simple HTTP Server (Python built-in)<br>
                        <strong>Port:</strong> 8000<br>
                        <strong>Status:</strong> Running ✅
                    </div>
                </div>
                
                <div class="section">
                    <h2>⚠️ Dependencies Required</h2>
                    <div class="status warning">
                        <p>To use the full AI features, you need to install dependencies:</p>
                        <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px;">
pip install fastapi uvicorn python-multipart
pip install torch diffusers transformers
pip install imageio opencv-python pillow
                        </pre>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔧 Next Steps</h2>
                    <ol>
                        <li><strong>Install Dependencies:</strong> Run the pip commands above</li>
                        <li><strong>Start Full Server:</strong> Run <code>python main.py</code> in the backend folder</li>
                        <li><strong>Or use batch file:</strong> Run <code>start_ai_video_simple.bat</code></li>
                    </ol>
                </div>
                
                <div class="section">
                    <h2>📁 Project Files</h2>
                    <div class="status info">
                        <strong>Generated Videos:</strong> """ + str(len(list(Path('outputs').glob('*.mp4')) if Path('outputs').exists() else [])) + """ files<br>
                        <strong>Uploaded Images:</strong> """ + str(len(list(Path('uploads').glob('*')) if Path('uploads').exists() else [])) + """ files<br>
                        <strong>Models Folder:</strong> """ + ("Exists" if Path('models').exists() else "Not created yet") + """
                    </div>
                </div>
                
                <div class="section">
                    <h2>🚀 Quick Install & Start</h2>
                    <div class="status success">
                        <p><strong>Run this command to install everything and start:</strong></p>
                        <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px;">
# In the ai-image-to-video folder:
install_dependencies.bat
                        </pre>
                    </div>
                </div>
            </body>
            </html>
            """
            
            self.wfile.write(html_content.encode())
            
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = {
                "server": "Simple HTTP Server",
                "dependencies_installed": False,
                "message": "Install FastAPI and other dependencies to use full features"
            }
            
            self.wfile.write(json.dumps(status).encode())
            
        else:
            # Serve static files
            super().do_GET()

def main():
    # Change to backend directory if we're not already there
    if not os.path.exists('main.py'):
        if os.path.exists('backend/main.py'):
            os.chdir('backend')
        else:
            print("❌ Error: Cannot find main.py")
            print("Please run this from the ai-image-to-video directory")
            return
    
    # Create necessary directories
    Path('outputs').mkdir(exist_ok=True)
    Path('uploads').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    print("🚀 Starting Simple AI Video Server...")
    print(f"📝 Server will be available at: http://localhost:{PORT}")
    print("💡 This is a minimal server - install dependencies for full features")
    print("🛑 Press Ctrl+C to stop")
    print()
    
    try:
        with socketserver.TCPServer(("", PORT), AIVideoHandler) as httpd:
            print(f"✅ Server started successfully on port {PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()
