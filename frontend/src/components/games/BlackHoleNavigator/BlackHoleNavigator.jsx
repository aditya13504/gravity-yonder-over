import React, { useState, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Trail, EffectComposer, Bloom } from '@react-three/drei';
import { 
  Box, 
  Slider, 
  Button, 
  Typography, 
  Paper, 
  Alert,
  FormControlLabel,
  Switch,
  Chip,
  CircularProgress
} from '@mui/material';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { useGameStore } from '../../../store/gameStore';
import { schwarzschildRadius, photonSphere } from '../../../physics/einstein';
import axios from 'axios';

// Black Hole Component with Accretion Disk
function BlackHole({ mass }) {
  const groupRef = useRef();
  const diskRef = useRef();
  
  useFrame((state) => {
    if (diskRef.current) {
      diskRef.current.rotation.z += 0.01;
    }
  });
  
  const rs = schwarzschildRadius(mass * 1.989e30) / 1e9; // km
  const photonR = photonSphere(mass * 1.989e30) / 1e9;
  
  return (
    <group ref={groupRef}>
      {/* Event Horizon */}
      <mesh>
        <sphereGeometry args={[rs / 1000, 32, 32]} />
        <meshBasicMaterial color="black" />
      </mesh>
      
      {/* Photon Sphere (glowing ring) */}
      <mesh>
        <torusGeometry args={[photonR / 1000, 0.01, 16, 100]} />
        <meshBasicMaterial color="#ffaa00" emissive="#ff6600" emissiveIntensity={2} />
      </mesh>
      
      {/* Accretion Disk */}
      <group ref={diskRef}>
        {[...Array(50)].map((_, i) => (
          <mesh key={i} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[(rs * (2 + i * 0.1)) / 1000, 0.02, 8, 50]} />
            <meshStandardMaterial
              color={new THREE.Color().setHSL(0.1 - i * 0.002, 1, 0.5)}
              emissive={new THREE.Color().setHSL(0.1 - i * 0.002, 1, 0.3)}
              emissiveIntensity={1 - i * 0.02}
              transparent
              opacity={0.8 - i * 0.015}
            />
          </mesh>
        ))}
      </group>
    </group>
  );
}

// Spaceship Component
function Spaceship({ initialPosition, blackHoleMass, targetPosition, onSuccess, onFailure }) {
  const meshRef = useRef();
  const [trajectory, setTrajectory] = useState([]);
  const [status, setStatus] = useState('flying');
  
  useFrame((state, delta) => {
    if (meshRef.current && status === 'flying') {
      const pos = meshRef.current.position;
      const vel = meshRef.current.userData.velocity || new THREE.Vector3(-2, 0, 0);
      
      // Calculate gravitational acceleration (simplified)
      const r = pos.length();
      const rs = schwarzschildRadius(blackHoleMass * 1.989e30) / 1e9 / 1000; // Convert to scene units
      
      if (r < rs) {
        setStatus('consumed');
        onFailure('Crossed event horizon!');
        return;
      }
      
      // Gravitational acceleration
      const G = 6.67430e-11;
      const M = blackHoleMass * 1.989e30;
      const accel = pos.clone().normalize().multiplyScalar(-G * M / (r * r * 1e18)); // Scale
      
      // Update velocity and position
      vel.add(accel.multiplyScalar(delta * 0.1));
      pos.add(vel.clone().multiplyScalar(delta * 0.1));
      
      meshRef.current.userData.velocity = vel;
      
      // Store trajectory
      setTrajectory(prev => [...prev.slice(-200), pos.clone()]);
      
      // Check if reached target
      if (pos.distanceTo(targetPosition) < 0.5) {
        setStatus('success');
        onSuccess();
      }
      
      // Check if escaped
      if (r > 50) {
        setStatus('escaped');
        onFailure('Escaped to infinity!');
      }
    }
  });
  
  return (
    <>
      <mesh ref={meshRef} position={initialPosition}>
        <coneGeometry args={[0.1, 0.3, 8]} />
        <meshStandardMaterial 
          color={status === 'flying' ? '#00ff00' : '#ff0000'} 
          emissive={status === 'flying' ? '#00ff00' : '#ff0000'}
          emissiveIntensity={0.5}
        />
      </mesh>
      
      {/* Trajectory trail */}
      {trajectory.length > 1 && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={trajectory.length}
              array={new Float32Array(trajectory.flatMap(p => [p.x, p.y, p.z]))}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#00ff00" opacity={0.5} transparent />
        </line>
      )}
    </>
  );
}

const BlackHoleNavigatorGame = () => {
  const [blackHoleMass, setBlackHoleMass] = useState(10); // Solar masses
  const [launchAngle, setLaunchAngle] = useState(45);
  const [launchVelocity, setLaunchVelocity] = useState(5);
  const [showRelativity, setShowRelativity] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [physicsData, setPhysicsData] = useState(null);
  const [educationalNotes, setEducationalNotes] = useState([]);
  const { addScore, updateGameProgress } = useGameStore();
  
  const targetPosition = new THREE.Vector3(10, 0, 0);
  
  const calculateSafeDistance = () => {
    const rs = schwarzschildRadius(blackHoleMass * 1.989e30) / 1e9; // km
    const photonR = photonSphere(blackHoleMass * 1.989e30) / 1e9;
    return { rs, photonR, safe: photonR * 2 };
  };
  
  // Run NVIDIA Modulus black hole navigation simulation
  const runModulusSimulation = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/games/black-hole-navigation', {
        black_hole_mass: blackHoleMass,
        launch_angle: launchAngle,
        launch_velocity: launchVelocity,
        show_relativity: showRelativity,
        target_position: [targetPosition.x, targetPosition.y, targetPosition.z],
        simulation_time: 200,
        time_steps: 100
      });
      
      if (response.data.success) {
        setPhysicsData(response.data.data);
        setEducationalNotes(response.data.educational_notes);
        setLaunching(true);
        addScore(response.data.score);
      } else {
        setError('Black hole navigation simulation failed');
      }
    } catch (err) {
      setError(`Failed to run simulation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const launchSpaceship = () => {
    runModulusSimulation();
  };
  
  const handleSuccess = () => {
    const score = Math.round(100 * (1 + blackHoleMass / 10));
    setResult({
      success: true,
      message: 'Navigation successful!',
      score
    });
    
    addScore(score);
    updateGameProgress('BlackHoleNavigator', {
      completed: true,
      bestScore: score
    });
    
    setLaunching(false);
  };
    const handleFailure = (reason) => {
    setResult({
      success: false,
      message: reason
    });
    setLaunching(false);
  };
  
  const resetSimulation = () => {
    setLaunching(false);
    setResult(null);
    setError(null);
    setPhysicsData(null);
    setEducationalNotes([]);
  };
  
  const { rs, photonR, safe } = calculateSafeDistance();
  
  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* 3D Scene */}
      <Box sx={{ flex: 1, background: 'black' }}>
        <Canvas camera={{ position: [15, 10, 15], fov: 60 }}>
          <ambientLight intensity={0.1} />
          <pointLight position={[10, 10, 10]} intensity={0.5} />
          <OrbitControls enablePan={false} />
          
          <EffectComposer>
            <Bloom luminanceThreshold={0.5} luminanceSmoothing={0.9} />
          </EffectComposer>
          
          {/* Black Hole */}
          <BlackHole mass={blackHoleMass} />
          
          {/* Target Position */}
          <mesh position={targetPosition}>
            <sphereGeometry args={[0.3, 16, 16]} />
            <meshStandardMaterial 
              color="#00ffff" 
              emissive="#00ffff" 
              emissiveIntensity={1}
              transparent
              opacity={0.8}
            />
          </mesh>
          
          {/* Safe zone indicator */}
          <mesh>
            <torusGeometry args={[safe / 1000, 0.02, 16, 100]} />
            <meshBasicMaterial color="green" transparent opacity={0.3} />
          </mesh>
          
          {/* Spaceship */}
          {launching && (
            <Spaceship
              initialPosition={new THREE.Vector3(
                10 * Math.cos(launchAngle * Math.PI / 180),
                10 * Math.sin(launchAngle * Math.PI / 180),
                0
              )}
              blackHoleMass={blackHoleMass}
              targetPosition={targetPosition}
              onSuccess={handleSuccess}
              onFailure={handleFailure}
            />
          )}
          
          {/* Info text */}
          <Text
            position={[0, -8, 0]}
            fontSize={0.5}
            color="white"
            anchorX="center"
          >
            Event Horizon: {rs.toFixed(2)} km | Photon Sphere: {photonR.toFixed(2)} km
          </Text>
        </Canvas>
      </Box>
      
      {/* Controls */}
      <Paper sx={{ width: 400, p: 3, m: 2, maxHeight: '90vh', overflow: 'auto' }}>        <Typography variant="h5" gutterBottom>
          ⚫ Black Hole Navigator - NVIDIA Modulus
        </Typography>
        
        <Alert severity="warning" sx={{ mb: 2 }}>
          Navigate your spacecraft near a black hole to reach the target without 
          crossing the event horizon! Real relativistic physics simulation.
        </Alert>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>
            Black Hole Mass: {blackHoleMass} Solar Masses
          </Typography>
          <Slider
            value={blackHoleMass}
            onChange={(e, v) => setBlackHoleMass(v)}
            min={1}
            max={50}
            disabled={launching}
          />
        </Box>
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>
            Launch Angle: {launchAngle}°
          </Typography>
          <Slider
            value={launchAngle}
            onChange={(e, v) => setLaunchAngle(v)}
            min={0}
            max={180}
            disabled={launching}
          />
        </Box>
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>
            Initial Velocity: {launchVelocity} c
          </Typography>
          <Slider
            value={launchVelocity}
            onChange={(e, v) => setLaunchVelocity(v)}
            min={0.1}
            max={10}
            step={0.1}
            disabled={launching}
          />
        </Box>
        
        <FormControlLabel
          control={
            <Switch
              checked={showRelativity}
              onChange={(e) => setShowRelativity(e.target.checked)}
            />
          }
          label="Show relativistic effects"
          sx={{ mb: 2 }}
        />
        
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Critical Distances:
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip 
              label={`Event Horizon: ${rs.toFixed(1)} km`} 
              color="error" 
              size="small" 
            />
            <Chip 
              label={`Photon Sphere: ${photonR.toFixed(1)} km`} 
              color="warning" 
              size="small" 
            />
            <Chip 
              label={`Safe Distance: ${safe.toFixed(1)} km`} 
              color="success" 
              size="small" 
            />
          </Box>
        </Box>
          <Button
          variant="contained"
          fullWidth
          onClick={launching ? resetSimulation : launchSpaceship}
          sx={{ mb: 2 }}
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : null}
        >
          {loading ? 'Computing...' : launching ? 'Reset' : 'Launch Spacecraft (Modulus)!'}
        </Button>
        
        {/* Educational Notes */}
        {educationalNotes.length > 0 && (
          <Paper sx={{ p: 2, mb: 3, bgcolor: 'info.main', color: 'info.contrastText' }}>
            <Typography variant="h6" gutterBottom>
              🎓 Black Hole Physics
            </Typography>
            {educationalNotes.map((note, index) => (
              <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                • {note}
              </Typography>
            ))}
          </Paper>
        )}
        
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Alert severity={result.success ? 'success' : 'error'}>
              <Typography variant="h6">{result.message}</Typography>
              {result.success && (
                <Typography>Score: +{result.score} points</Typography>
              )}
            </Alert>
          </motion.div>
        )}
        
        <Box sx={{ mt: 3, p: 2, backgroundColor: 'background.default', borderRadius: 1 }}>
          <Typography variant="subtitle2" gutterBottom>
            Physics Tips:
          </Typography>
          <Typography variant="body2" color="text.secondary">
            • The event horizon is the point of no return
          </Typography>
          <Typography variant="body2" color="text.secondary">
            • Light can orbit at the photon sphere
          </Typography>
          <Typography variant="body2" color="text.secondary">
            • Time dilation increases near the black hole
          </Typography>
          <Typography variant="body2" color="text.secondary">
            • Use gravitational slingshot to your advantage
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default BlackHoleNavigatorGame;