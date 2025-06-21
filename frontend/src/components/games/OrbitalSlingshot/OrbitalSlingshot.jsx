import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Sphere } from '@react-three/drei';
import { Box, Slider, Button, Typography, Paper, Alert, CircularProgress, Switch, FormControlLabel } from '@mui/material';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { useGameStore } from '../../../store/gameStore';
import axios from 'axios';

// Spacecraft Component with real-time physics
function Spacecraft({ position, trail }) {
  const meshRef = useRef();
  
  return (
    <>
      <mesh ref={meshRef} position={position}>
        <coneGeometry args={[0.3, 1, 8]} />
        <meshStandardMaterial color="silver" />
      </mesh>
      
      {/* Trail */}
      {trail && trail.length > 1 && (
        <Line
          points={trail}
          color="cyan"
          lineWidth={2}
        />
      )}
    </>
  );
}

// Planetary bodies component
function PlanetaryBodies({ bodies }) {
  return (
    <>
      {bodies.map((body, index) => (
        <group key={index}>
          <Sphere 
            position={body.position} 
            args={[body.radius]}
          >
            <meshStandardMaterial color={body.color} />
          </Sphere>
          <Text
            position={[body.position[0], body.position[1] + body.radius + 1, body.position[2]]}
            fontSize={0.5}
            color="white"
            anchorX="center"
            anchorY="middle"
          >
            {body.name}
          </Text>
        </group>
      ))}
    </>
  );
}

// Velocity vector visualization
function VelocityVector({ position, velocity, scale = 0.1 }) {
  if (!velocity || velocity.length === 0) return null;
  
  const endPosition = [
    position[0] + velocity[0] * scale,
    position[1] + velocity[1] * scale,
    position[2] + velocity[2] * scale
  ];
  
  return (
    <Line
      points={[position, endPosition]}
      color="yellow"
      lineWidth={3}
    />
  );
}

const OrbitalSlingshotGame = () => {
  const [launchVelocity, setLaunchVelocity] = useState(8.0);
  const [launchAngle, setLaunchAngle] = useState(45);
  const [targetPlanet, setTargetPlanet] = useState('Mars');
  const [isSimulating, setIsSimulating] = useState(false);
  const [physicsData, setPhysicsData] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showVelocityVector, setShowVelocityVector] = useState(true);
  const [results, setResults] = useState(null);
  const [educationalNotes, setEducationalNotes] = useState([]);
  const [spacecraftTrail, setSpacecraftTrail] = useState([]);
  const { addScore } = useGameStore();
  const animationRef = useRef();
  
  // Planetary bodies configuration
  const planetaryBodies = [
    { name: 'Earth', position: [0, 0, 0], radius: 2, color: 'blue', mass: 5.972e24 },
    { name: 'Moon', position: [15, 0, 0], radius: 0.5, color: 'gray', mass: 7.342e22 },
    { name: 'Mars', position: [30, 5, 0], radius: 1.5, color: 'red', mass: 6.39e23 },
    { name: 'Jupiter', position: [60, 10, 0], radius: 4, color: 'orange', mass: 1.898e27 }
  ];
  
  // Run NVIDIA Modulus orbital mechanics simulation
  const runModulusSimulation = async () => {
    setLoading(true);
    setError(null);
    setCurrentFrame(0);
    setSpacecraftTrail([]);
    
    try {
      const response = await axios.post('/api/games/orbital-slingshot', {
        launch_velocity: launchVelocity,
        launch_angle: launchAngle,
        target_planet: targetPlanet,
        planetary_bodies: planetaryBodies,
        simulation_time: 1000, // seconds
        time_steps: 200
      });
      
      if (response.data.success) {
        setPhysicsData(response.data.data);
        setEducationalNotes(response.data.educational_notes);
        setIsSimulating(true);
        addScore(response.data.score);
      } else {
        setError('Orbital simulation failed');
      }
    } catch (err) {
      setError(`Failed to run orbital simulation: ${err.message}`);
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
            setSpacecraftTrail(prev => [...prev.slice(-50), currentPos]); // Keep last 50 positions
            
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
  }, [isSimulating, physicsData]);
  
  const calculateResults = useCallback(() => {
    if (!physicsData) return;
    
    const finalPosition = physicsData.positions[physicsData.positions.length - 1];
    const finalVelocity = physicsData.velocities[physicsData.velocities.length - 1];
    const totalTime = physicsData.times[physicsData.times.length - 1];
    
    // Check if spacecraft reached target vicinity
    const targetBody = planetaryBodies.find(body => body.name === targetPlanet);
    const distanceToTarget = Math.sqrt(
      Math.pow(finalPosition[0] - targetBody.position[0], 2) +
      Math.pow(finalPosition[1] - targetBody.position[1], 2) +
      Math.pow(finalPosition[2] - targetBody.position[2], 2)
    );
    
    const success = distanceToTarget < targetBody.radius * 3; // Within 3 radii counts as success
    
    setResults({
      success,
      finalPosition: finalPosition.map(pos => pos.toFixed(2)),
      finalVelocity: Math.sqrt(finalVelocity[0]**2 + finalVelocity[1]**2 + finalVelocity[2]**2).toFixed(2),
      totalTime: totalTime.toFixed(2),
      distanceToTarget: distanceToTarget.toFixed(2),
      targetRadius: (targetBody.radius * 3).toFixed(2)
    });
  }, [physicsData, targetPlanet, planetaryBodies]);
  
  const resetSimulation = () => {
    setIsSimulating(false);
    setPhysicsData(null);
    setCurrentFrame(0);
    setSpacecraftTrail([]);
    setResults(null);
    setError(null);
    setEducationalNotes([]);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
  };
  
  // Get current spacecraft position and velocity for rendering
  const getCurrentPosition = () => {
    if (!physicsData || !isSimulating) return [0, 0, 5]; // Launch position
    return physicsData.positions[currentFrame] || [0, 0, 5];
  };
  
  const getCurrentVelocity = () => {
    if (!physicsData || !isSimulating) return [0, 0, 0];
    return physicsData.velocities[currentFrame] || [0, 0, 0];
  };
  
  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* 3D Scene */}
      <Box sx={{ flex: 1 }}>
        <Canvas camera={{ position: [40, 30, 40], fov: 60 }}>
          <ambientLight intensity={0.3} />
          <directionalLight position={[50, 50, 25]} intensity={0.8} />
          <pointLight position={[0, 0, 0]} intensity={1} color="white" />
          
          <OrbitControls enablePan={true} />
          
          {/* Planetary Bodies */}
          <PlanetaryBodies bodies={planetaryBodies} />
          
          {/* Spacecraft */}
          <Spacecraft 
            position={getCurrentPosition()} 
            trail={spacecraftTrail}
          />
          
          {/* Velocity Vector */}
          {showVelocityVector && (
            <VelocityVector 
              position={getCurrentPosition()} 
              velocity={getCurrentVelocity()}
              scale={0.5}
            />
          )}
          
          {/* Grid */}
          <gridHelper args={[100, 20]} position={[0, -5, 0]} />
        </Canvas>
      </Box>
      
      {/* Controls */}
      <Paper sx={{ width: 380, p: 3, m: 2, overflow: 'auto' }}>
        <Typography variant="h5" gutterBottom>
          🚀 Orbital Slingshot - NVIDIA Modulus Physics
        </Typography>
        
        <Typography variant="body2" sx={{ mb: 3 }}>
          Use gravitational slingshot maneuvers to reach your target planet! 
          Real NVIDIA Modulus n-body orbital mechanics simulation.
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>Launch Velocity: {launchVelocity} km/s</Typography>
          <Slider
            value={launchVelocity}
            onChange={(e, v) => setLaunchVelocity(v)}
            min={5}
            max={15}
            step={0.1}
            disabled={isSimulating || loading}
          />
        </Box>
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>Launch Angle: {launchAngle}°</Typography>
          <Slider
            value={launchAngle}
            onChange={(e, v) => setLaunchAngle(v)}
            min={0}
            max={90}
            step={1}
            disabled={isSimulating || loading}
          />
        </Box>
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>Target Planet:</Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {['Moon', 'Mars', 'Jupiter'].map((planet) => (
              <Button
                key={planet}
                variant={targetPlanet === planet ? 'contained' : 'outlined'}
                size="small"
                onClick={() => setTargetPlanet(planet)}
                disabled={isSimulating || loading}
              >
                {planet}
              </Button>
            ))}
          </Box>
        </Box>
        
        <FormControlLabel
          control={
            <Switch
              checked={showVelocityVector}
              onChange={(e) => setShowVelocityVector(e.target.checked)}
            />
          }
          label="Show Velocity Vector"
          sx={{ mb: 2 }}
        />
        
        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <Button
            variant="contained"
            fullWidth
            onClick={runModulusSimulation}
            disabled={isSimulating || loading}
            startIcon={loading ? <CircularProgress size={20} /> : null}
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
        
        {/* Educational Notes */}
        {educationalNotes.length > 0 && (
          <Paper sx={{ p: 2, mb: 3, bgcolor: 'info.main', color: 'info.contrastText' }}>
            <Typography variant="h6" gutterBottom>
              🎓 Orbital Mechanics Insights
            </Typography>
            {educationalNotes.map((note, index) => (
              <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                • {note}
              </Typography>
            ))}
          </Paper>
        )}
        
        {/* Real-time status */}
        {isSimulating && physicsData && (
          <Paper sx={{ p: 2, mb: 3, bgcolor: 'primary.main', color: 'primary.contrastText' }}>
            <Typography variant="h6" gutterBottom>
              🎯 Mission Status
            </Typography>
            <Typography variant="body2">
              Time: {physicsData.times[currentFrame]?.toFixed(1) || 0}s
            </Typography>
            <Typography variant="body2">
              Frame: {currentFrame + 1} / {physicsData.positions?.length || 0}
            </Typography>
            <Typography variant="body2">
              Velocity: {getCurrentVelocity().map(v => v.toFixed(2)).join(', ')} km/s
            </Typography>
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
              bgcolor: results.success ? 'success.main' : 'warning.main',
              color: results.success ? 'success.contrastText' : 'warning.contrastText'
            }}>
              <Typography variant="h6" gutterBottom>
                {results.success ? '🎉 Mission Success!' : '❌ Mission Failed'}
              </Typography>
              <Typography variant="body2">Final Position: [{results.finalPosition.join(', ')}]</Typography>
              <Typography variant="body2">Final Velocity: {results.finalVelocity} km/s</Typography>
              <Typography variant="body2">Mission Time: {results.totalTime}s</Typography>
              <Typography variant="body2">
                Distance to {targetPlanet}: {results.distanceToTarget} units 
                (Target: &lt; {results.targetRadius})
              </Typography>
            </Paper>
          </motion.div>
        )}
        
        <Typography variant="caption" sx={{ mt: 2, display: 'block' }}>
          Powered by NVIDIA Modulus N-Body Orbital Mechanics
        </Typography>
      </Paper>
    </Box>
  );
};

export default OrbitalSlingshotGame;
