import React, { useState, useCallback, useEffect, useRef } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import './App.css';

const API_URL = 'http://localhost:8001';
const WS_URL = 'ws://localhost:8001';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
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
    duration: 4,
    fps: 7,
    motionBucketId: 127,
    noiseAugStrength: 0.02
  });

  React.useEffect(() => {
    fetchGPUInfo();
  }, []);

  useEffect(() => {
    if (jobId && loading) {
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
            setError('Video generation failed');
          }
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      return () => {
        if (wsRef.current) {
          wsRef.current.close();
        }
      };
    }
  }, [jobId, loading]);

  const fetchGPUInfo = async () => {
    try {
      const response = await axios.get(`${API_URL}/gpu-info`);
      setGpuInfo(response.data);
    } catch (err) {
      console.error('Failed to fetch GPU info:', err);
    }
  };

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage({
          file: file,
          preview: e.target.result
        });
        setVideoUrl(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    maxFiles: 1
  });

  const downloadGeneratedVideo = async (jobId) => {
    try {
      const response = await axios.get(`${API_URL}/download/${jobId}`, {
        responseType: 'blob'
      });
      
      const videoBlob = new Blob([response.data], { type: 'video/mp4' });
      const url = URL.createObjectURL(videoBlob);
      setVideoUrl(url);
    } catch (err) {
      setError('Failed to download video');
    }
  };

  const generateVideo = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);
    setProgress({ progress: 0, message: 'Initializing...', eta: null });

    const formData = new FormData();
    formData.append('file', selectedImage.file);
    formData.append('duration', settings.duration);
    formData.append('fps', settings.fps);
    formData.append('motion_bucket_id', settings.motionBucketId);
    formData.append('noise_aug_strength', settings.noiseAugStrength);

    try {
      const response = await axios.post(`${API_URL}/generate-video`, formData);
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
        <h1>AI Image to Video Generator</h1>
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
          <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
            <input {...getInputProps()} />
            {selectedImage ? (
              <img src={selectedImage.preview} alt="Selected" className="preview-image" />
            ) : (
              <div className="dropzone-content">
                <p>Drag & drop an image here, or click to select</p>
                <p className="file-types">Supports: JPG, PNG, GIF, BMP, WebP</p>
              </div>
            )}
          </div>

          <div className="settings">
            <h3>Video Settings</h3>
            <div className="setting-group">
              <label>
                Duration (seconds):
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={settings.duration}
                  onChange={(e) => setSettings({...settings, duration: parseInt(e.target.value)})}
                />
              </label>
            </div>
            <div className="setting-group">
              <label>
                Motion Intensity:
                <input
                  type="range"
                  min="1"
                  max="255"
                  value={settings.motionBucketId}
                  onChange={(e) => setSettings({...settings, motionBucketId: parseInt(e.target.value)})}
                />
                <span>{settings.motionBucketId}</span>
              </label>
            </div>
            <div className="setting-group">
              <label>
                Noise Strength:
                <input
                  type="range"
                  min="0"
                  max="0.1"
                  step="0.01"
                  value={settings.noiseAugStrength}
                  onChange={(e) => setSettings({...settings, noiseAugStrength: parseFloat(e.target.value)})}
                />
                <span>{settings.noiseAugStrength.toFixed(2)}</span>
              </label>
            </div>
          </div>

          <button
            className="generate-btn"
            onClick={generateVideo}
            disabled={!selectedImage || loading}
          >
            {loading ? 'Generating...' : 'Generate Video'}
          </button>

          {error && <div className="error">{error}</div>}

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