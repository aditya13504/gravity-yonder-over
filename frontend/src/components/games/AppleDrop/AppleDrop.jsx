import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Sky, Text } from '@react-three/drei';
import { Box, Slider, Button, Typography, Paper, Alert, CircularProgress } from '@mui/material';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { useGameStore } from '../../../store/gameStore';
import axios from 'axios';

// Apple Component with real-time physics data
function Apple({ physicsData, isSimulating, onComplete }) {
  const meshRef = useRef();
  const animationRef = useRef();
  
  useEffect(() => {
    if (isSimulating && physicsData && meshRef.current) {
      let currentIndex = 0;
      const positions = physicsData.positions;
      const times = physicsData.times;
      
      const animate = () => {
        if (currentIndex < positions.length && meshRef.current) {
          const height = Math.max(0, positions[currentIndex]);
          meshRef.current.position.y = height;
          currentIndex++;
          
          if (height > 0) {
            animationRef.current = requestAnimationFrame(animate);
          } else {
            onComplete();
          }
        }
      };
      
      animate();
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isSimulating, physicsData, onComplete]);
  
  const initialHeight = physicsData ? physicsData.positions[0] : 10;
  
  return (
    <mesh ref={meshRef} position={[0, initialHeight, 0]}>
      <sphereGeometry args={[0.3, 32, 32]} />
      <meshStandardMaterial color="red" />
    </mesh>
  );
}

// Tree Component
function Tree() {
  return (
    <group position={[0, 0, 0]}>
      {/* Trunk */}
      <mesh position={[0, 2, 0]}>
        <cylinderGeometry args={[0.5, 0.7, 4]} />
        <meshStandardMaterial color="#8B4513" />
      </mesh>
      {/* Leaves */}
      <mesh position={[0, 5, 0]}>
        <sphereGeometry args={[3, 16, 16]} />
        <meshStandardMaterial color="green" />
      </mesh>
    </group>
  );
}

// Ground Component
function Ground() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
      <planeGeometry args={[100, 100]} />
      <meshStandardMaterial color="#567d46" />
    </mesh>
  );
}

const AppleDropGame = () => {
  const [height, setHeight] = useState(10);
  const [gravity, setGravity] = useState(9.81);
  const [isSimulating, setIsSimulating] = useState(false);
  const [physicsData, setPhysicsData] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [educationalNotes, setEducationalNotes] = useState([]);
  const { addScore } = useGameStore();
  
  // Run real-time NVIDIA Modulus simulation
  const runModulusSimulation = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/games/apple-drop', {
        height: height,
        gravity: gravity,
        time_steps: 100
      });
      
      if (response.data.success) {
        setPhysicsData(response.data.data);
        setEducationalNotes(response.data.educational_notes);
        setIsSimulating(true);
        addScore(response.data.score);
      } else {
        setError('Simulation failed');
      }
    } catch (err) {
      setError(`Failed to run simulation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const handleSimulationComplete = () => {
    setIsSimulating(false);
    if (physicsData) {
      const finalVelocity = physicsData.velocities[physicsData.velocities.length - 1];
      const timeToFall = physicsData.times[physicsData.times.length - 1];
      const kineticEnergy = 0.5 * 0.2 * finalVelocity * finalVelocity; // Assuming 0.2kg apple
      
      setResults({
        timeToFall: timeToFall.toFixed(2),
        finalVelocity: finalVelocity.toFixed(2),
        kineticEnergy: kineticEnergy.toFixed(2),
      });
    }
  };
  
  const resetSimulation = () => {
    setIsSimulating(false);
    setPhysicsData(null);
    setResults(null);
    setError(null);
    setEducationalNotes([]);
  };
  
  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* 3D Scene */}
      <Box sx={{ flex: 1 }}>
        <Canvas camera={{ position: [15, 10, 15], fov: 60 }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <Sky />
          <OrbitControls enablePan={false} />
          
          <Tree />
          <Ground />
          <Apple 
            physicsData={physicsData} 
            isSimulating={isSimulating} 
            onComplete={handleSimulationComplete}
          />
          
          <Text
            position={[0, height + 2, 0]}
            fontSize={0.5}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            {height.toFixed(1)}m
          </Text>
        </Canvas>
      </Box>
      
      {/* Controls */}
      <Paper sx={{ width: 350, p: 3, m: 2 }}>
        <Typography variant="h5" gutterBottom>
          🍎 Apple Drop Experiment - Real NVIDIA Modulus Physics
        </Typography>
        
        <Typography variant="body2" sx={{ mb: 3 }}>
          Drop an apple using real-time NVIDIA Modulus physics simulations. 
          This uses GPU-accelerated calculations for accurate gravitational motion.
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>Height: {height}m</Typography>
          <Slider
            value={height}
            onChange={(e, v) => setHeight(v)}
            min={1}
            max={50}
            disabled={isSimulating || loading}
          />
        </Box>
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>Gravity: {gravity} m/s²</Typography>
          <Slider
            value={gravity}
            onChange={(e, v) => setGravity(v)}
            min={1}
            max={20}
            step={0.1}
            disabled={isSimulating || loading}
          />
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <Button
            variant="contained"
            fullWidth
            onClick={runModulusSimulation}
            disabled={isSimulating || loading}
            startIcon={loading ? <CircularProgress size={20} /> : null}
          >
            {loading ? 'Computing...' : 'Drop Apple (Modulus)!'}
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
            <Paper sx={{ p: 2, bgcolor: 'success.main', color: 'success.contrastText' }}>
              <Typography variant="h6" gutterBottom>
                📊 Results (NVIDIA Modulus)
              </Typography>
              <Typography>Fall Time: {results.timeToFall}s</Typography>
              <Typography>Final Velocity: {results.finalVelocity} m/s</Typography>
              <Typography>Kinetic Energy: {results.kineticEnergy} J</Typography>
            </Paper>
          </motion.div>
        )}
        
        <Typography variant="caption" sx={{ mt: 2, display: 'block' }}>
          Powered by NVIDIA Modulus Physics Engine for educational accuracy
        </Typography>
      </Paper>
    </Box>
  );
};

export default AppleDropGame;