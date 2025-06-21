import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere } from '@react-three/drei';
import { Box, Slider, Button, Typography, Paper, Alert, CircularProgress, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { useGameStore } from '../../../store/gameStore';
import axios from 'axios';

// Spacecraft Component with real-time physics
function Spacecraft({ position, trail, escaped }) {
  const meshRef = useRef();
  
  useFrame(() => {
    if (meshRef.current && escaped) {
      meshRef.current.rotation.y += 0.1;
    }
  });
  
  return (
    <>
      <mesh ref={meshRef} position={position}>
        <coneGeometry args={[0.2, 0.8, 8]} />
        <meshStandardMaterial color={escaped ? "gold" : "silver"} />
      </mesh>
      
      {/* Exhaust trail */}
      {trail && trail.length > 1 && (
        <Line
          points={trail}
          color={escaped ? "gold" : "cyan"}
          lineWidth={escaped ? 3 : 2}
        />
      )}
    </>
  );
}

// Planet Component
function Planet({ radius, color, name, position = [0, 0, 0] }) {
  const meshRef = useRef();
  
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01;
    }
  });
  
  return (
    <group position={position}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 32, 32]} />
        <meshStandardMaterial color={color} />
      </mesh>
      
      {/* Atmosphere glow */}
      <mesh>
        <sphereGeometry args={[radius * 1.1, 32, 32]} />
        <meshBasicMaterial 
          color={color} 
          transparent={true} 
          opacity={0.2}
        />
      </mesh>
      
      <Text
        position={[0, radius + 2, 0]}
        fontSize={0.8}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {name}
      </Text>
    </group>
  );
}

// Escape velocity visualization circles
function EscapeZone({ radius, visible }) {
  if (!visible) return null;
  
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[radius * 0.98, radius * 1.02, 64]} />
      <meshBasicMaterial color="yellow" transparent={true} opacity={0.3} />
    </mesh>
  );
}

const EscapeVelocityGame = () => {
  const [launchVelocity, setLaunchVelocity] = useState(8.0);
  const [launchAngle, setLaunchAngle] = useState(90); // Vertical launch
  const [selectedPlanet, setSelectedPlanet] = useState('Earth');
  const [isSimulating, setIsSimulating] = useState(false);
  const [physicsData, setPhysicsData] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [educationalNotes, setEducationalNotes] = useState([]);
  const [spacecraftTrail, setSpacecraftTrail] = useState([]);
  const [escaped, setEscaped] = useState(false);
  const { addScore } = useGameStore();
  const animationRef = useRef();
  
  // Planet configurations with realistic escape velocities
  const planets = {
    'Earth': {
      radius: 3,
      color: 'blue',
      mass: 5.972e24,
      escapeVelocity: 11.2,
      surfaceGravity: 9.81
    },
    'Mars': {
      radius: 2,
      color: 'red',
      mass: 6.39e23,
      escapeVelocity: 5.0,
      surfaceGravity: 3.71
    },
    'Moon': {
      radius: 1.5,
      color: 'gray',
      mass: 7.342e22,
      escapeVelocity: 2.4,
      surfaceGravity: 1.62
    },
    'Jupiter': {
      radius: 8,
      color: 'orange',
      mass: 1.898e27,
      escapeVelocity: 59.5,
      surfaceGravity: 24.79
    }
  };
  
  const currentPlanet = planets[selectedPlanet];
  
  // Run NVIDIA Modulus escape velocity simulation
  const runModulusSimulation = async () => {
    setLoading(true);
    setError(null);
    setCurrentFrame(0);
    setSpacecraftTrail([]);
    setEscaped(false);
    
    try {
      const response = await axios.post('/api/games/escape-velocity', {
        launch_velocity: launchVelocity,
        launch_angle: launchAngle,
        planet: selectedPlanet,
        planet_data: currentPlanet,
        simulation_time: 300, // seconds
        time_steps: 150
      });
      
      if (response.data.success) {
        setPhysicsData(response.data.data);
        setEducationalNotes(response.data.educational_notes);
        setIsSimulating(true);
        addScore(response.data.score);
      } else {
        setError('Escape velocity simulation failed');
      }
    } catch (err) {
      setError(`Failed to run escape simulation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Animation loop for spacecraft movement
  useEffect(() => {
    if (isSimulating && physicsData) {
      const animate = () => {
        setCurrentFrame(prevFrame => {
          const nextFrame = prevFrame + 1;
          
          if (nextFrame < physicsData.positions.length) {
            // Update trail
            const currentPos = physicsData.positions[nextFrame];
            setSpacecraftTrail(prev => [...prev.slice(-30), currentPos]); // Keep last 30 positions
            
            // Check if spacecraft has escaped
            const distanceFromCenter = Math.sqrt(
              currentPos[0]**2 + currentPos[1]**2 + currentPos[2]**2
            );
            
            const escapeRadius = currentPlanet.radius * 10; // Escape zone
            if (distanceFromCenter > escapeRadius && !escaped) {
              setEscaped(true);
            }
            
            animationRef.current = requestAnimationFrame(animate);
            return nextFrame;
          } else {
            // Simulation complete
            setIsSimulating(false);
            calculateResults();
            return prevFrame;
          }
        });
      };
      
      animationRef.current = requestAnimationFrame(animate);
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isSimulating, physicsData, escaped, currentPlanet.radius]);
  
  const calculateResults = useCallback(() => {
    if (!physicsData) return;
    
    const finalPosition = physicsData.positions[physicsData.positions.length - 1];
    const finalVelocity = physicsData.velocities[physicsData.velocities.length - 1];
    const maxDistance = Math.max(...physicsData.positions.map(pos => 
      Math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    ));
    
    const escapeRadius = currentPlanet.radius * 10;
    const successfulEscape = maxDistance > escapeRadius;
    
    const finalSpeed = Math.sqrt(
      finalVelocity[0]**2 + finalVelocity[1]**2 + finalVelocity[2]**2
    );
    
    setResults({
      success: successfulEscape,
      maxDistance: maxDistance.toFixed(2),
      finalSpeed: finalSpeed.toFixed(2),
      escapeVelocityRequired: currentPlanet.escapeVelocity,
      launchVelocityUsed: launchVelocity,
      escapeRadius: escapeRadius.toFixed(2),
      totalTime: physicsData.times[physicsData.times.length - 1].toFixed(2)
    });
  }, [physicsData, currentPlanet, launchVelocity]);
  
  const resetSimulation = () => {
    setIsSimulating(false);
    setPhysicsData(null);
    setCurrentFrame(0);
    setSpacecraftTrail([]);
    setResults(null);
    setError(null);
    setEducationalNotes([]);
    setEscaped(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };
  
  // Get current spacecraft position for rendering
  const getCurrentPosition = () => {
    if (!physicsData || !isSimulating) {
      return [0, currentPlanet.radius + 0.5, 0]; // Just above surface
    }
    return physicsData.positions[currentFrame] || [0, currentPlanet.radius + 0.5, 0];
  };
  
  const getCurrentVelocity = () => {
    if (!physicsData || !isSimulating) return [0, 0, 0];
    return physicsData.velocities[currentFrame] || [0, 0, 0];
  };
  
  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* 3D Scene */}
      <Box sx={{ flex: 1 }}>
        <Canvas camera={{ position: [20, 15, 20], fov: 60 }}>
          <ambientLight intensity={0.4} />
          <directionalLight position={[30, 30, 15]} intensity={0.8} />
          <pointLight position={[0, 0, 0]} intensity={2} color="white" />
          
          <OrbitControls enablePan={true} />
          
          {/* Planet */}
          <Planet 
            radius={currentPlanet.radius}
            color={currentPlanet.color}
            name={selectedPlanet}
          />
          
          {/* Escape zone visualization */}
          <EscapeZone 
            radius={currentPlanet.radius * 10}
            visible={true}
          />
          
          {/* Spacecraft */}
          <Spacecraft 
            position={getCurrentPosition()} 
            trail={spacecraftTrail}
            escaped={escaped}
          />
          
          {/* Reference grid */}
          <gridHelper args={[100, 20]} position={[0, -currentPlanet.radius - 2, 0]} />
          
          {/* Stars background */}
          {Array.from({ length: 100 }, (_, i) => (
            <mesh key={i} position={[
              (Math.random() - 0.5) * 200,
              (Math.random() - 0.5) * 200,
              (Math.random() - 0.5) * 200
            ]}>
              <sphereGeometry args={[0.1]} />
              <meshBasicMaterial color="white" />
            </mesh>
          ))}
        </Canvas>
      </Box>
      
      {/* Controls */}
      <Paper sx={{ width: 400, p: 3, m: 2, overflow: 'auto' }}>
        <Typography variant="h5" gutterBottom>
          🚀 Escape Velocity Challenge - NVIDIA Modulus
        </Typography>
        
        <Typography variant="body2" sx={{ mb: 3 }}>
          Can you launch a spacecraft fast enough to escape the planet's gravity? 
          Real NVIDIA Modulus gravitational physics simulation.
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Planet</InputLabel>
          <Select
            value={selectedPlanet}
            onChange={(e) => setSelectedPlanet(e.target.value)}
            disabled={isSimulating || loading}
          >
            {Object.keys(planets).map(planet => (
              <MenuItem key={planet} value={planet}>
                {planet} (Escape: {planets[planet].escapeVelocity} km/s)
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>
            Launch Velocity: {launchVelocity} km/s
            {launchVelocity >= currentPlanet.escapeVelocity && 
              <span style={{color: 'green'}}> ✓ Above escape velocity!</span>
            }
          </Typography>
          <Slider
            value={launchVelocity}
            onChange={(e, v) => setLaunchVelocity(v)}
            min={1}
            max={Math.max(currentPlanet.escapeVelocity * 2, 20)}
            step={0.1}
            disabled={isSimulating || loading}
            marks={[
              { value: currentPlanet.escapeVelocity, label: `Escape: ${currentPlanet.escapeVelocity}` }
            ]}
          />
        </Box>
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>Launch Angle: {launchAngle}°</Typography>
          <Slider
            value={launchAngle}
            onChange={(e, v) => setLaunchAngle(v)}
            min={0}
            max={90}
            step={5}
            disabled={isSimulating || loading}
            marks={[
              { value: 0, label: '0° (Horizontal)' },
              { value: 45, label: '45°' },
              { value: 90, label: '90° (Vertical)' }
            ]}
          />
        </Box>
        
        <Paper sx={{ p: 2, mb: 3, bgcolor: 'info.light' }}>
          <Typography variant="h6" gutterBottom>
            📊 {selectedPlanet} Data
          </Typography>
          <Typography variant="body2">Escape Velocity: {currentPlanet.escapeVelocity} km/s</Typography>
          <Typography variant="body2">Surface Gravity: {currentPlanet.surfaceGravity} m/s²</Typography>
          <Typography variant="body2">Your Velocity: {launchVelocity} km/s</Typography>
        </Paper>
        
        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <Button
            variant="contained"
            fullWidth
            onClick={runModulusSimulation}
            disabled={isSimulating || loading}
            startIcon={loading ? <CircularProgress size={20} /> : null}
            color={launchVelocity >= currentPlanet.escapeVelocity ? 'success' : 'primary'}
          >
            {loading ? 'Computing...' : 'Launch!'}
          </Button>
          <Button
            variant="outlined"
            fullWidth
            onClick={resetSimulation}
            disabled={loading}
          >
            Reset
          </Button>
        </Box>
        
        {/* Real-time status */}
        {isSimulating && physicsData && (
          <Paper sx={{ p: 2, mb: 3, bgcolor: escaped ? 'success.main' : 'primary.main', color: 'white' }}>
            <Typography variant="h6" gutterBottom>
              🎯 Mission Status {escaped && '🎉'}
            </Typography>
            <Typography variant="body2">
              Time: {physicsData.times[currentFrame]?.toFixed(1) || 0}s
            </Typography>
            <Typography variant="body2">
              Altitude: {(Math.sqrt(
                getCurrentPosition()[0]**2 + 
                getCurrentPosition()[1]**2 + 
                getCurrentPosition()[2]**2
              ) - currentPlanet.radius).toFixed(2)} units
            </Typography>
            <Typography variant="body2">
              Speed: {Math.sqrt(
                getCurrentVelocity()[0]**2 + 
                getCurrentVelocity()[1]**2 + 
                getCurrentVelocity()[2]**2
              ).toFixed(2)} km/s
            </Typography>
            {escaped && (
              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                🌟 ESCAPED GRAVITY WELL!
              </Typography>
            )}
          </Paper>
        )}
        
        {/* Educational Notes */}
        {educationalNotes.length > 0 && (
          <Paper sx={{ p: 2, mb: 3, bgcolor: 'info.main', color: 'info.contrastText' }}>
            <Typography variant="h6" gutterBottom>
              🎓 Physics Insights
            </Typography>
            {educationalNotes.map((note, index) => (
              <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                • {note}
              </Typography>
            ))}
          </Paper>
        )}
        
        {results && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Paper sx={{ 
              p: 2, 
              bgcolor: results.success ? 'success.main' : 'error.main',
              color: 'white'
            }}>
              <Typography variant="h6" gutterBottom>
                {results.success ? '🎉 Escape Successful!' : '❌ Failed to Escape'}
              </Typography>
              <Typography variant="body2">
                Max Distance: {results.maxDistance} units
              </Typography>
              <Typography variant="body2">
                Escape Zone: {results.escapeRadius} units
              </Typography>
              <Typography variant="body2">
                Final Speed: {results.finalSpeed} km/s
              </Typography>
              <Typography variant="body2">
                Mission Time: {results.totalTime}s
              </Typography>
              <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic' }}>
                Required: {results.escapeVelocityRequired} km/s | Used: {results.launchVelocityUsed} km/s
              </Typography>
            </Paper>
          </motion.div>
        )}
        
        <Typography variant="caption" sx={{ mt: 2, display: 'block' }}>
          Powered by NVIDIA Modulus Gravitational Physics Engine
        </Typography>
      </Paper>
    </Box>
  );
};

export default EscapeVelocityGame;
