import React, { useState, useRef, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Grid, Stars } from '@react-three/drei';
import { Box, Paper, Typography, Button, TextField, IconButton } from '@mui/material';
import { Delete as DeleteIcon, PlayArrow, Pause, Refresh } from '@mui/icons-material';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';
import ParameterControls from './ParameterControls';

// Celestial Body Component
function CelestialBody({ body, onSelect, selected }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  
  useFrame(() => {
    if (meshRef.current && body.trail) {
      // Update trail - implementation depends on trail system
    }
  });
  
  return (
    <mesh
      ref={meshRef}
      position={[body.position.x, body.position.y, body.position.z]}
      onClick={() => onSelect(body.id)}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <sphereGeometry args={[body.radius, 32, 32]} />
      <meshStandardMaterial
        color={body.color}
        emissive={body.color}
        emissiveIntensity={selected || hovered ? 0.5 : 0.2}
      />
    </mesh>
  );
}

// Gravity Simulation Component
function GravitySimulation({ bodies, running, dt }) {
  const G = 6.67430e-11;
  
  useFrame((state, delta) => {
    if (!running || bodies.length < 2) return;
    
    // Calculate forces between all bodies
    const forces = bodies.map(() => new THREE.Vector3());
    
    for (let i = 0; i < bodies.length; i++) {
      for (let j = i + 1; j < bodies.length; j++) {
        const body1 = bodies[i];
        const body2 = bodies[j];
        
        const r = body2.position.clone().sub(body1.position);
        const distance = r.length();
        
        if (distance > 0) {
          const forceMagnitude = G * body1.mass * body2.mass / (distance * distance);
          const forceDirection = r.normalize();
          const force = forceDirection.multiplyScalar(forceMagnitude);
          
          forces[i].add(force);
          forces[j].sub(force);
        }
      }
    }
    
    // Update velocities and positions
    bodies.forEach((body, i) => {
      const acceleration = forces[i].divideScalar(body.mass);
      body.velocity.add(acceleration.multiplyScalar(dt));
      body.position.add(body.velocity.clone().multiplyScalar(dt));
      
      // Add to trail
      if (body.trail) {
        body.trail.push(body.position.clone());
        if (body.trail.length > 500) {
          body.trail.shift();
        }
      }
    });
  });
  
  return null;
}

const GravityPlayground = () => {
  const [bodies, setBodies] = useState([]);
  const [selectedBody, setSelectedBody] = useState(null);
  const [running, setRunning] = useState(false);
  const [newBodyForm, setNewBodyForm] = useState({
    name: '',
    mass: 1,
    x: 0,
    y: 0,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    radius: 1,
    color: '#4169E1',
  });
  
  const addBody = () => {
    const body = {
      id: Date.now(),
      name: newBodyForm.name || `Body ${bodies.length + 1}`,
      mass: newBodyForm.mass * 5.972e24, // Earth masses to kg
      position: new THREE.Vector3(
        newBodyForm.x * 1.496e11,
        newBodyForm.y * 1.496e11,
        newBodyForm.z * 1.496e11
      ),
      velocity: new THREE.Vector3(
        newBodyForm.vx * 1000,
        newBodyForm.vy * 1000,
        newBodyForm.vz * 1000
      ),
      radius: newBodyForm.radius,
      color: newBodyForm.color,
      trail: [],
    };
    
    setBodies([...bodies, body]);
    setNewBodyForm({
      ...newBodyForm,
      name: '',
    });
  };
  
  const removeBody = (id) => {
    setBodies(bodies.filter(b => b.id !== id));
    if (selectedBody?.id === id) {
      setSelectedBody(null);
    }
  };
  
  const clearAll = () => {
    setBodies([]);
    setSelectedBody(null);
    setRunning(false);
  };
  
  const toggleSimulation = () => {
    setRunning(!running);
  };
  
  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* 3D Scene */}
      <Box sx={{ flex: 1 }}>
        <Canvas camera={{ position: [50, 30, 50], fov: 60 }}>
          <ambientLight intensity={0.1} />
          <pointLight position={[0, 0, 0]} intensity={1} />
          <Stars radius={300} depth={50} count={5000} factor={4} fade />
          <OrbitControls />
          <Grid args={[200, 200]} position={[0, -50, 0]} />
          
          <GravitySimulation bodies={bodies} running={running} dt={3600} />
          
          {bodies.map((body) => (
            <CelestialBody
              key={body.id}
              body={body}
              selected={selectedBody?.id === body.id}
              onSelect={(id) => setSelectedBody(bodies.find(b => b.id === id))}
            />
          ))}
        </Canvas>
      </Box>
      
      {/* Controls */}
      <Box sx={{ width: 400, display: 'flex', flexDirection: 'column' }}>
        <Paper sx={{ p: 2, m: 2, flex: 1, overflow: 'auto' }}>
          <Typography variant="h5" gutterBottom>
            🔬 Gravity Sandbox
          </Typography>
          
          {/* Simulation Controls */}
          <Box sx={{ display: 'flex', gap: 1, mb: 3 }}>
            <Button
              variant="contained"
              startIcon={running ? <Pause /> : <PlayArrow />}
              onClick={toggleSimulation}
              disabled={bodies.length < 2}
            >
              {running ? 'Pause' : 'Run'}
            </Button>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={clearAll}
            >
              Clear All
            </Button>
          </Box>
          
          {/* Add Body Form */}
          <ParameterControls
            parameters={newBodyForm}
            onChange={setNewBodyForm}
            onAdd={addBody}
          />
          
          {/* Bodies List */}
          <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
            Celestial Bodies ({bodies.length})
          </Typography>
          
          <AnimatePresence>
            {bodies.map((body) => (
              <motion.div
                key={body.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
              >
                <Paper
                  sx={{
                    p: 1,
                    mb: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    backgroundColor: selectedBody?.id === body.id ? 'action.selected' : 'background.paper',
                    cursor: 'pointer',
                  }}
                  onClick={() => setSelectedBody(body)}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box
                      sx={{
                        width: 20,
                        height: 20,
                        borderRadius: '50%',
                        backgroundColor: body.color,
                      }}
                    />
                    <Typography>{body.name}</Typography>
                  </Box>
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeBody(body.id);
                    }}
                  >
                    <DeleteIcon />
                  </IconButton>
                </Paper>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {/* Selected Body Info */}
          {selectedBody && (
            <Paper sx={{ p: 2, mt: 3, backgroundColor: 'primary.dark' }}>
              <Typography variant="h6" gutterBottom>
                {selectedBody.name}
              </Typography>
              <Typography variant="body2">
                Mass: {(selectedBody.mass / 5.972e24).toFixed(2)} Earth masses
              </Typography>
              <Typography variant="body2">
                Position: ({(selectedBody.position.x / 1.496e11).toFixed(2)}, 
                {(selectedBody.position.y / 1.496e11).toFixed(2)}, 
                {(selectedBody.position.z / 1.496e11).toFixed(2)}) AU
              </Typography>
              <Typography variant="body2">
                Velocity: {(selectedBody.velocity.length() / 1000).toFixed(2)} km/s
              </Typography>
            </Paper>
          )}
        </Paper>
      </Box>
    </Box>
  );
};

export default GravityPlayground;