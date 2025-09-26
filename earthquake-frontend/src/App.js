import React, { useState, useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

// We will load Leaflet from a CDN, so we don't import it here.

function App() {
  const [formData, setFormData] = useState({
    latitude: '21.7679', // Default to a central location in India
    longitude: '78.8718',
    depth: '',
    mag: '',
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Refs for chart and map
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const [leafletLoaded, setLeafletLoaded] = useState(false);

  // Effect to load Leaflet CSS and JS from CDN
  useEffect(() => {
    // Load CSS
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
    link.integrity = 'sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=';
    link.crossOrigin = '';
    document.head.appendChild(link);

    // Load JS
    const script = document.createElement('script');
    script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
    script.integrity = 'sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=';
    script.crossOrigin = '';
    script.onload = () => {
      setLeafletLoaded(true); // Set state to true when script is loaded
    };
    document.body.appendChild(script);

    return () => {
      document.head.removeChild(link);
      document.body.removeChild(script);
    };
  }, []);


  // Effect for fetching graph data and initializing the line chart
  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/graph-data');
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        if (chartInstance.current) {
          chartInstance.current.destroy();
        }

        if (chartRef.current) {
            const ctx = chartRef.current.getContext('2d');
            chartInstance.current = new Chart(ctx, {
              type: 'line',
              data: {
                labels: data.labels,
                datasets: [{
                  label: 'Magnitude',
                  data: data.values,
                  backgroundColor: 'rgba(63, 81, 181, 0.2)',
                  borderColor: 'rgba(63, 81, 181, 1)',
                  borderWidth: 2, fill: true, tension: 0.1
                }]
              },
              options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: true, text: 'Magnitude of Last 100 Earthquakes', font: { size: 16 }}},
                scales: {
                  y: { beginAtZero: false, title: { display: true, text: 'Magnitude' }},
                  x: { title: { display: true, text: 'Recent Earthquake Sequence' }, ticks: { display: false }}
                }
              }
            });
        }
      } catch (error) { console.error("Failed to fetch graph data:", error); }
    };
    fetchGraphData();

    // Correct cleanup for the chart
    return () => {
        if(chartInstance.current) {
            chartInstance.current.destroy();
        }
    }
  }, []);

  // Effect for initializing the map and fetching recent earthquake data
  useEffect(() => {
    if (leafletLoaded && mapRef.current && !mapInstance.current) {
        const L = window.L; // Leaflet is now available on the window object
        
        // Fix for default icon issue with Leaflet
        delete L.Icon.Default.prototype._getIconUrl;
        L.Icon.Default.mergeOptions({
            iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
            iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
            shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
        });

        mapInstance.current = L.map(mapRef.current).setView([20.5937, 78.9629], 5);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(mapInstance.current);
        
        // Fetch and display markers
        const fetchMapData = async () => {
          try {
            const response = await fetch('http://127.0.0.1:8000/recent-earthquakes');
            const quakes = await response.json();
            if (quakes.error) throw new Error(quakes.error);
            
            quakes.forEach(quake => {
              L.circleMarker([quake.latitude, quake.longitude], {
                radius: quake.mag * 1.5,
                fillColor: quake.mag > 5 ? "#c62828" : "#fb8c00",
                color: "#000",
                weight: 1,
                opacity: 1,
                fillOpacity: 0.7
              }).bindPopup(`Magnitude: ${quake.mag}`).addTo(mapInstance.current);
            });

          } catch (error) { console.error("Failed to fetch map data:", error); }
        };

        fetchMapData();
    }
    
    // Correct cleanup for the map
    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, [leafletLoaded]); // This effect depends on leafletLoaded


  const handleChange = (e) => setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    const payload = { latitude: parseFloat(formData.latitude), longitude: parseFloat(formData.longitude) };
    if (formData.depth) payload.depth = parseFloat(formData.depth);
    if (formData.mag) payload.mag = parseFloat(formData.mag);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      setResult({ error: 'Failed to connect to the prediction API.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>{`
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f0f2f5; color: #333; margin: 0; padding: 1rem; }
        .main-container { display: grid; grid-template-columns: 1fr; gap: 1.5rem; max-width: 1400px; margin: auto; }
        @media (min-width: 1024px) { .main-container { grid-template-columns: 350px 1fr; } .bottom-row { grid-column: 2 / 3; } }
        .form-container { background-color: white; border-radius: 12px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1); padding: 2rem; grid-row: 1 / 3; }
        h2 { text-align: center; color: #1a237e; margin-top: 0; margin-bottom: 1.5rem; }
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; font-weight: 600; margin-bottom: 0.5rem; color: #555; }
        .form-group input { width: 100%; padding: 0.75rem; border: 1px solid #ccc; border-radius: 8px; box-sizing: border-box; transition: all 0.2s; }
        .form-group input:focus { outline: none; border-color: #3f51b5; box-shadow: 0 0 0 3px rgba(63, 81, 181, 0.2); }
        .btn-submit { width: 100%; padding: 0.8rem; background-color: #3f51b5; color: white; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; transition: all 0.3s; }
        .btn-submit:disabled { background-color: #9fa8da; cursor: not-allowed; }
        .result-card { padding: 1.5rem; border-radius: 12px; background-color: #fafafa; margin-top: 1.5rem; }
        .result-prediction { font-size: 1.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem; }
        .prediction-significant { color: #c62828; } .prediction-weak { color: #2e7d32; }
        .error { color: #c62828; font-weight: bold; text-align: center; }
        .info-banner { background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 1rem; margin-top: 1rem; border-radius: 4px; font-size: 0.9rem; }
        .visuals-container { display: grid; grid-template-columns: 1fr; grid-template-rows: 1fr 1fr; gap: 1.5rem; }
        @media (min-width: 768px) { .visuals-container { grid-template-columns: 1fr 1fr; grid-template-rows: 1fr; } }
        .viz-card { background-color: white; border-radius: 12px; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1); padding: 1rem; display: flex; flex-direction: column; }
        .chart-wrapper, .map-wrapper { position: relative; width: 100%; height: 350px; }
      `}</style>
      <div className="main-container">
        <div className="form-container">
          <h2>🌍 Earthquake Prediction</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group"><label>Latitude *</label><input type="number" name="latitude" value={formData.latitude} onChange={handleChange} step="any" required /></div>
            <div className="form-group"><label>Longitude *</label><input type="number" name="longitude" value={formData.longitude} onChange={handleChange} step="any" required /></div>
            <div className="form-group"><label>Depth (km)</label><input type="number" name="depth" value={formData.depth} onChange={handleChange} step="any" placeholder="Optional, uses average" /></div>
            <div className="form-group"><label>Magnitude</label><input type="number" name="mag" value={formData.mag} onChange={handleChange} step="any" placeholder="Optional, uses average" /></div>
            <button type="submit" disabled={loading} className="btn-submit">{loading ? 'Analyzing...' : 'Predict Risk'}</button>
          </form>
          {result && (
            <div className="result-card">
              <h3>Prediction Result</h3>
              {result.error ? <p className="error">{result.error}</p> : (
                <>
                  <p className={`result-prediction ${result.prediction === 'Significant' ? 'prediction-significant' : 'prediction-weak'}`}>{result.prediction} Risk</p>
                  <p><strong>Confidence:</strong> {result.confidence}</p>
                  {result.used_avg_values && <div className="info-banner"><p>Used historical averages for prediction.</p></div>}
                </>
              )}
            </div>
          )}
        </div>
        <div className="visuals-container">
            <div className="viz-card">
                <div className="chart-wrapper"><canvas ref={chartRef}></canvas></div>
            </div>
            <div className="viz-card">
                <div id="map" className="map-wrapper" ref={mapRef}></div>
            </div>
        </div>
      </div>
    </>
  );
}

export default App;

