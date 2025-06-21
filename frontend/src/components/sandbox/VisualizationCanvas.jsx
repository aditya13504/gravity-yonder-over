import React, { useRef, useEffect } from 'react';

const VisualizationCanvas = ({ params, isRunning, data }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = 800;
    canvas.height = 600;

    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw coordinate system
      drawCoordinateSystem(ctx, canvas.width, canvas.height);
      
      // Draw celestial bodies
      drawCelestialBodies(ctx, params);
      
      // Draw force vectors if enabled
      if (params.showForceVectors) {
        drawForceVectors(ctx, params);
      }
      
      // Draw trails if enabled
      if (params.showTrails && data.length > 0) {
        drawTrails(ctx, data);
      }

      if (isRunning) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    if (isRunning) {
      animate();
    } else {
      // Draw static state
      animate();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [params, isRunning, data]);

  const drawCoordinateSystem = (ctx, width, height) => {
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    
    // Draw grid
    for (let x = 0; x <= width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    for (let y = 0; y <= height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  };

  const drawCelestialBodies = (ctx, params) => {
    const centerX = 400;
    const centerY = 300;
    
    // Draw central mass (larger body)
    ctx.fillStyle = '#ff6b35';
    ctx.beginPath();
    ctx.arc(centerX, centerY, Math.sqrt(params.mass1) / 10, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw orbiting body
    ctx.fillStyle = '#4ecdc4';
    const orbitX = centerX + params.distance / 5;
    const orbitY = centerY;
    ctx.beginPath();
    ctx.arc(orbitX, orbitY, Math.sqrt(params.mass2) / 5, 0, 2 * Math.PI);
    ctx.fill();
  };

  const drawForceVectors = (ctx, params) => {
    // Calculate and draw gravitational force vectors
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    
    const centerX = 400;
    const centerY = 300;
    const orbitX = centerX + params.distance / 5;
    const orbitY = centerY;
    
    // Draw force vector from orbiting body toward central mass
    ctx.beginPath();
    ctx.moveTo(orbitX, orbitY);
    ctx.lineTo(centerX, centerY);
    ctx.stroke();
    
    // Draw arrowhead
    const angle = Math.atan2(centerY - orbitY, centerX - orbitX);
    const arrowLength = 10;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(
      centerX - arrowLength * Math.cos(angle - 0.3),
      centerY - arrowLength * Math.sin(angle - 0.3)
    );
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(
      centerX - arrowLength * Math.cos(angle + 0.3),
      centerY - arrowLength * Math.sin(angle + 0.3)
    );
    ctx.stroke();
  };

  const drawTrails = (ctx, data) => {
    if (data.length < 2) return;
    
    ctx.strokeStyle = '#4ecdc4';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.7;
    
    ctx.beginPath();
    ctx.moveTo(data[0].x, data[0].y);
    
    for (let i = 1; i < data.length; i++) {
      ctx.lineTo(data[i].x, data[i].y);
    }
    
    ctx.stroke();
    ctx.globalAlpha = 1.0;
  };

  return (
    <div className="visualization-canvas">
      <canvas ref={canvasRef} />
      <div className="canvas-info">
        <p>Red circle: Central mass ({params.mass1} kg)</p>
        <p>Blue circle: Orbiting body ({params.mass2} kg)</p>
        <p>Red line: Gravitational force vector</p>
      </div>
    </div>
  );
};

export default VisualizationCanvas;
