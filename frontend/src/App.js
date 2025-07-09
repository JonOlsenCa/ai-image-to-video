import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [textPrompt, setTextPrompt] = useState('');
  const [videoUrl, setVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gpuInfo, setGpuInfo] = useState(null);
  const [progress, setProgress] = useState({
    progress: 0,
    message: '',
    eta: null
  });
  const [jobId, setJobId] = useState(null);
  const wsRef = useRef(null);
  const [settings, setSettings] = useState({
    duration: 10
  });

  React.useEffect(() => {
    fetchGPUInfo();
  }, []);

  useEffect(() => {
    if (jobId && loading) {
      // Set a timeout to check if video is ready
      const timeoutId = setTimeout(() => {
        console.log('Checking if video is ready after timeout...');
        downloadGeneratedVideo(jobId);
      }, 15000); // Check after 15 seconds
      
      // Connect to WebSocket
      wsRef.current = new WebSocket(`${WS_URL}/ws/${jobId}`);
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'progress') {
          setProgress({
            progress: data.data.progress,
            message: data.data.message,
            eta: data.data.eta
          });
          
          // Check if completed
          if (data.data.status === 'completed') {
            setLoading(false);
            // Download the video
            downloadGeneratedVideo(jobId);
          } else if (data.data.status === 'failed') {
            setLoading(false);
            setError(data.data.message || 'Video generation failed');
          }
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Fallback to polling if WebSocket fails
        startPolling(jobId);
      };
      
      wsRef.current.onclose = () => {
        console.log('WebSocket closed');
        // If still loading, fallback to polling
        if (loading) {
          startPolling(jobId);
        }
      };
      
      return () => {
        clearTimeout(timeoutId);
        if (wsRef.current) {
          wsRef.current.close();
        }
      };
    }
  }, [jobId, loading]);

  const startPolling = (jobId) => {
    console.log('Falling back to polling for job status');
    const pollInterval = setInterval(async () => {
      try {
        const response = await axios.get(`${API_URL}/job-status/${jobId}`);
        if (response.data.status === 'completed') {
          clearInterval(pollInterval);
          setLoading(false);
          downloadGeneratedVideo(jobId);
        } else if (response.data.status === 'failed') {
          clearInterval(pollInterval);
          setLoading(false);
          setError('Video generation failed');
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 2000);
    
    // Clean up after 2 minutes
    setTimeout(() => clearInterval(pollInterval), 120000);
  };

  const fetchGPUInfo = async () => {
    try {
      const response = await axios.get(`${API_URL}/gpu-info`);
      setGpuInfo(response.data);
    } catch (err) {
      console.error('Failed to fetch GPU info:', err);
    }
  };


  const downloadGeneratedVideo = async (jobId) => {
    try {
      // First check if video exists
      const statusResponse = await axios.get(`${API_URL}/job-status/${jobId}`);
      console.log('Job status:', statusResponse.data);
      
      if (statusResponse.data.status === 'completed' || statusResponse.data.status === 'not_found') {
        // Try to download the video
        const response = await axios.get(`${API_URL}/download/${jobId}`, {
          responseType: 'blob'
        });
        
        const videoBlob = new Blob([response.data], { type: 'video/mp4' });
        const url = URL.createObjectURL(videoBlob);
        setVideoUrl(url);
        setLoading(false);
      } else if (statusResponse.data.status === 'processing') {
        // Still processing, wait a bit more
        console.log('Video still processing...');
        setTimeout(() => downloadGeneratedVideo(jobId), 2000);
      } else {
        setError('Video generation failed or not found');
        setLoading(false);
      }
    } catch (err) {
      console.error('Download error:', err);
      if (err.response?.status === 404) {
        // Video not ready yet, try again
        setTimeout(() => downloadGeneratedVideo(jobId), 2000);
      } else {
        setError('Failed to download video');
        setLoading(false);
      }
    }
  };

  const handleImageSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const generateVideo = async () => {
    if (!textPrompt.trim() && !selectedImage) {
      setError('Please provide a text prompt or select an image');
      return;
    }

    setLoading(true);
    setError(null);
    setProgress({ progress: 0, message: 'Initializing...', eta: null });

    try {
      const formData = new FormData();
      if (selectedImage) {
        formData.append('image', selectedImage);
      }
      
      // Use query params for prompt and duration
      const params = new URLSearchParams({
        prompt: textPrompt || 'animate this image with subtle motion',
        duration: settings.duration
      });
      
      const response = await axios.post(`${API_URL}/generate-video?${params}`, formData, {
        headers: selectedImage ? {
          'Content-Type': 'multipart/form-data'
        } : {}
      });
      
      console.log('Video generation started:', response.data);
      setJobId(response.data.job_id);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start video generation');
      setLoading(false);
    }
  };

  const downloadVideo = () => {
    if (!videoUrl) return;
    
    const a = document.createElement('a');
    a.href = videoUrl;
    a.download = 'generated-video.mp4';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const formatETA = (seconds) => {
    if (!seconds || seconds < 0) return '';
    
    if (seconds < 60) {
      return `${Math.round(seconds)}s remaining`;
    }
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s remaining`;
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Image & Text to Video Generator</h1>
        {gpuInfo && (
          <div className="gpu-info">
            {gpuInfo.gpu_available ? (
              <span className="gpu-available">
                GPU: {gpuInfo.gpu_name} ({gpuInfo.gpu_memory})
              </span>
            ) : (
              <span className="gpu-unavailable">No GPU detected</span>
            )}
          </div>
        )}
      </header>

      <main className="App-main">
        <div className="upload-section">
          <div className="input-options">
            <div className="image-upload-section">
              <h3>Option 1: Upload an Image</h3>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="file-input"
                id="image-input"
              />
              <label htmlFor="image-input" className="file-label">
                Choose Image
              </label>
              {imagePreview && (
                <div className="image-preview">
                  <img src={imagePreview} alt="Preview" />
                </div>
              )}
            </div>
            
            <div className="divider">AND/OR</div>
            
            <div className="text-input-section">
              <h3>Option 2: Describe Your Video</h3>
              <textarea
                className="text-prompt"
                placeholder="Type EXACTLY one of these animations:&#10;&#10;pan left to right&#10;zoom out&#10;rotate&#10;shake&#10;tilt&#10;fade"
                value={textPrompt}
                onChange={(e) => setTextPrompt(e.target.value)}
                rows={4}
              />
              <div className="animation-categories">
                <div className="animation-group ai-powered">
                  <h4>🤖 AI Person Animation</h4>
                  <div className="animation-buttons">
                    <button type="button" onClick={() => setTextPrompt('person walking')} className="anim-btn ai-btn">🚶 Walking</button>
                    <button type="button" onClick={() => setTextPrompt('person dancing')} className="anim-btn ai-btn">💃 Dancing</button>
                    <button type="button" onClick={() => setTextPrompt('person waving')} className="anim-btn ai-btn">👋 Waving</button>
                    <button type="button" onClick={() => setTextPrompt('person jumping')} className="anim-btn ai-btn">🏃 Jumping</button>
                  </div>
                </div>
                
                <div className="animation-group ai-powered">
                  <h4>😊 AI Facial Animation</h4>
                  <div className="animation-buttons">
                    <button type="button" onClick={() => setTextPrompt('person smiling')} className="anim-btn ai-btn">😊 Smiling</button>
                    <button type="button" onClick={() => setTextPrompt('person talking')} className="anim-btn ai-btn">💬 Talking</button>
                    <button type="button" onClick={() => setTextPrompt('person nodding')} className="anim-btn ai-btn">✅ Nodding</button>
                    <button type="button" onClick={() => setTextPrompt('person looking around')} className="anim-btn ai-btn">👀 Looking</button>
                  </div>
                </div>
                
                <div className="animation-group">
                  <h4>📹 Camera Movement</h4>
                  <div className="animation-buttons">
                    <button type="button" onClick={() => setTextPrompt('zoom in')} className="anim-btn">🔍 Zoom In</button>
                    <button type="button" onClick={() => setTextPrompt('zoom out')} className="anim-btn">🔍 Zoom Out</button>
                    <button type="button" onClick={() => setTextPrompt('pan left')} className="anim-btn">⬅️ Pan Left</button>
                    <button type="button" onClick={() => setTextPrompt('pan right')} className="anim-btn">➡️ Pan Right</button>
                  </div>
                </div>
                
                <div className="animation-group">
                  <h4>🌿 Object Animation</h4>
                  <div className="animation-buttons">
                    <button type="button" onClick={() => setTextPrompt('leaves rustling')} className="anim-btn ai-btn">🍃 Leaves</button>
                    <button type="button" onClick={() => setTextPrompt('water flowing')} className="anim-btn ai-btn">💧 Water</button>
                    <button type="button" onClick={() => setTextPrompt('fire flickering')} className="anim-btn ai-btn">🔥 Fire</button>
                    <button type="button" onClick={() => setTextPrompt('clouds moving')} className="anim-btn ai-btn">☁️ Clouds</button>
                  </div>
                </div>
              </div>
              
              <div className="animation-summary">
                <p><strong>🚀 AI-Powered Animations:</strong></p>
                <p className="animation-list">
                  <span className="ai-features">✨ Real person movement • Facial expressions • Object physics • Advanced AI</span>
                </p>
                <p className="fallback-note">
                  <small>💡 Falls back to simple animations if AI models aren't available</small>
                </p>
              </div>
              <p className="text-help">
                💡 Click a button above or type the exact phrase for best results!
              </p>
            </div>
          </div>

          <div className="settings">
            <h3>Video Settings</h3>
            <div className="setting-group">
              <label>
                Duration (seconds):
                <input
                  type="number"
                  min="6"
                  max="30"
                  step="6"
                  value={settings.duration}
                  onChange={(e) => setSettings({...settings, duration: parseInt(e.target.value)})}
                />
              </label>
              <p className="setting-help">
                Each 2-second clip takes ~30-60 seconds to generate
              </p>
            </div>
          </div>

          <button
            className="generate-btn"
            onClick={generateVideo}
            disabled={loading || (!textPrompt.trim() && !selectedImage)}
          >
            {loading ? 'Generating...' : 'Generate Video'}
          </button>

          {error && (
            <div className="error">
              <strong>Error:</strong> {error}
              <br />
              <small>Note: Currently using test video generator. Real AI models will be loaded after successful testing.</small>
            </div>
          )}

          {loading && (
            <div className="progress-container">
              <div className="progress-bar-wrapper">
                <div className="progress-bar">
                  <div 
                    className="progress-bar-fill" 
                    style={{ width: `${progress.progress}%` }}
                  />
                </div>
                <div className="progress-info">
                  <span className="progress-percentage">{Math.round(progress.progress)}%</span>
                  <span className="progress-eta">{formatETA(progress.eta)}</span>
                </div>
              </div>
              <div className="progress-message">{progress.message}</div>
            </div>
          )}
        </div>

        {videoUrl && (
          <div className="video-section">
            <h3>Generated Video</h3>
            <video controls src={videoUrl} className="generated-video" />
            <button className="download-btn" onClick={downloadVideo}>
              Download Video
            </button>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;