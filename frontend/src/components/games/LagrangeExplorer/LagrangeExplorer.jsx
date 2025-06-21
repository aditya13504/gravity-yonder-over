import React, { useState, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line, Html } from '@react-three/drei';
import {
  Box,
  Slider,
  Button,
  Typography,
  Paper,
  ToggleButton,
  ToggleButtonGroup,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  CircularProgress
} from '@mui/material';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { useGameStore } from '../../../store/gameStore';
import axios from 'axios';

// Calculate Lagrange points for two-body system
function calculateLagrangePoints(m1, m2, distance) {
  const mu = m2 / (m1 + m2);
  const points = [];
  
  // L1 - Between the bodies
  const L1_x = distance * (1 - Math.pow(mu / 3, 1/3));
  points.push({ name: 'L1', position: new THREE.Vector3(L1_x, 0, 0) });
  
  // L2 - Beyond secondary body
  const L2_x = distance * (1 + Math.pow(mu / 3, 1/3));
  points.push({ name: 'L2', position: new THREE.Vector3(L2_x, 0, 0) });
  
  // L3 - Opposite side of primary
  const L3_x = -distance * (1 - 5 * mu / 12);
  points.push({ name: 'L3', position: new THREE.Vector3(L3_x, 0, 0) });
  
  // L4 - 60 degrees ahead
  const L4_x = distance * Math.cos(Math.PI / 3) - distance * mu;
  const L4_y = distance * Math.sin(Math.PI / 3);
  points.push({ name: 'L4', position: new THREE.Vector3(L4_x, L4_y, 0) });
  
  // L5 - 60 degrees behind
  const L5_x = distance * Math.cos(Math.PI / 3) - distance * mu;
  const L5_y = -distance * Math.sin(Math.PI / 3);
  points.push({ name: 'L5', position: new THREE.Vector3(L5_x, L5_y, 0) });
  
  return points;
}

// Binary System Component
function BinarySystem({ primary, secondary, distance, showOrbits }) {
  const groupRef = useRef();
  const primaryRef = useRef();
  const secondaryRef = useRef();
  
  useFrame((state) => {
    if (groupRef.current) {
      const t = state.clock.getElapsedTime() * 0.5;
      
      // Calculate positions based on center of mass
      const totalMass = primary.mass + secondary.mass;
      const r1 = distance * secondary.mass / totalMass;
      const r2 = distance * primary.mass / totalMass;
      
      if (primaryRef.current) {
        primaryRef.current.position.x = -r1 * Math.cos(t);
        primaryRef.current.position.y = -r1 * Math.sin(t);
      }
      
      if (secondaryRef.current) {
        secondaryRef.current.position.x = r2 * Math.cos(t);
        secondaryRef.current.position.y = r2 * Math.sin(t);
      }
    }
  });
  
  return (
    <group ref={groupRef}>
      {/* Primary body */}
      <mesh ref={primaryRef}>
        <sphereGeometry args={[primary.radius, 32, 32]} />
        <meshStandardMaterial 
          color={primary.color} 
          emissive={primary.color} 
          emissiveIntensity={0.3}
        />
        <Html distanceFactor={10}>
          <div style={{ color: 'white', fontSize: '12px' }}>{primary.name}</div>
        </Html>
      </mesh>
      
      {/* Secondary body */}
      <mesh ref={secondaryRef}>
        <sphereGeometry args={[secondary.radius, 32, 32]} />
        <meshStandardMaterial 
          color={secondary.color}
          emissive={secondary.color} 
          emissiveIntensity={0.2}
        />
        <Html distanceFactor={10}>
          <div style={{ color: 'white', fontSize: '12px' }}>{secondary.name}</div>
        </Html>
      </mesh>
      
      {/* Orbit paths */}
      {showOrbits && (
        <>
          <Line
            points={Array.from({ length: 64 }, (_, i) => {
              const angle = (i / 63) * Math.PI * 2;
              const r = distance * secondary.mass / (primary.mass + secondary.mass);
              return new THREE.Vector3(
                r * Math.cos(angle),
                r * Math.sin(angle),
                0
              );
            })}
            color="rgba(255,255,255,0.3)"
            lineWidth={1}
          />
          <Line
            points={Array.from({ length: 64 }, (_, i) => {
              const angle = (i / 63) * Math.PI * 2;
              const r = distance * primary.mass / (primary.mass + secondary.mass);
              return new THREE.Vector3(
                r * Math.cos(angle),
                r * Math.sin(angle),
                0
              );
            })}
            color="rgba(255,255,255,0.3)"
            lineWidth={1}
          />
        </>
      )}
    </group>
  );
}

// Lagrange Point Marker
function LagrangePointMarker({ point, isTarget, isOccupied, onSelect }) {
  const [hovered, setHovered] = useState(false);
  
  return (
    <group position={point.position}>
      <mesh
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        onClick={() => onSelect(point)}
      >
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial
          color={isTarget ? '#00ff00' : isOccupied ? '#ff0000' : '#ffff00'}
          emissive={isTarget ? '#00ff00' : isOccupied ? '#ff0000' : '#ffff00'}
          emissiveIntensity={hovered ? 1 : 0.5}
          transparent
          opacity={0.8}
        />
      </mesh>
      <Html distanceFactor={10}>
        <div style={{ 
          color: 'white', 
          fontSize: '14px',
          fontWeight: 'bold',
          textShadow: '0 0 4px black'
        }}>
          {point.name}
        </div>
      </Html>
    </group>
  );
}

// Satellite Component
function Satellite({ position, lagrangePoint, onReachTarget }) {
  const meshRef = useRef();
  const [reached, setReached] = useState(false);
  
  useFrame(() => {
    if (meshRef.current && lagrangePoint && !reached) {
      const currentPos = meshRef.current.position;
      const targetPos = lagrangePoint.position;
      
      // Animate towards target
      currentPos.lerp(targetPos, 0.02);
      
      // Check if reached
      if (currentPos.distanceTo(targetPos) < 0.3) {
        setReached(true);
        onReachTarget();
      }
    }
  });
  
  return (
    <mesh ref={meshRef} position={position}>
      <boxGeometry args={[0.15, 0.15, 0.15]} />
      <meshStandardMaterial 
        color="#00ffff" 
        emissive="#00ffff" 
        emissiveIntensity={0.5}
      />
    </mesh>
  );
}

const LagrangeExplorerGame = () => {
  const [system, setSystem] = useState('earth-moon');
  const [targetPoint, setTargetPoint] = useState('L4');
  const [showGravityField, setShowGravityField] = useState(false);
  const [showOrbits, setShowOrbits] = useState(true);
  const [deploying, setDeploying] = useState(false);
  const [deployedSatellites, setDeployedSatellites] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [physicsData, setPhysicsData] = useState(null);
  const [educationalNotes, setEducationalNotes] = useState([]);
  const { addScore, updateGameProgress } = useGameStore();
  
  const systems = {
    'earth-moon': {
      primary: { name: 'Earth', mass: 5.972e24, radius: 0.5, color: '#4169E1' },
      secondary: { name: 'Moon', mass: 7.342e22, radius: 0.2, color: '#C0C0C0' },
      distance: 10
    },
    'sun-earth': {
      primary: { name: 'Sun', mass: 1.989e30, radius: 1, color: '#FDB813' },
      secondary: { name: 'Earth', mass: 5.972e24, radius: 0.3, color: '#4169E1' },
      distance: 15
    },
    'jupiter-io': {
      primary: { name: 'Jupiter', mass: 1.898e27, radius: 0.8, color: '#DAA520' },
      secondary: { name: 'Io', mass: 8.93e22, radius: 0.15, color: '#FFDEAD' },
      distance: 8
    }
  };
  
  const currentSystem = systems[system];
  const lagrangePoints = calculateLagrangePoints(
    currentSystem.primary.mass,
    currentSystem.secondary.mass,
    currentSystem.distance
  );
  
  // Run NVIDIA Modulus Lagrange point simulation
  const runModulusSimulation = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post('/api/games/lagrange-points', {
        system: system,
        system_data: currentSystem,
        target_point: targetPoint,
        simulation_time: 100,
        time_steps: 50
      });
      
      if (response.data.success) {
        setPhysicsData(response.data.data);
        setEducationalNotes(response.data.educational_notes);
        addScore(response.data.score);
      } else {
        setError('Lagrange point simulation failed');
      }
    } catch (err) {
      setError(`Failed to run simulation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const deploySatellite = () => {
    const target = lagrangePoints.find(p => p.name === targetPoint);
    if (!target) return;
    
    setDeploying(true);
    
    // Add satellite
    const satellite = {
      id: Date.now(),
      startPosition: new THREE.Vector3(0, -5, 0),
      targetPoint: target
    };
    
    setDeployedSatellites([...deployedSatellites, satellite]);
    
    // Run Modulus simulation for validation
    runModulusSimulation();
  };
  
  const handleSatelliteReachTarget = (satelliteId) => {
    const satellite = deployedSatellites.find(s => s.id === satelliteId);
    if (!satellite) return;
    
    const isCorrect = satellite.targetPoint.name === targetPoint;
    const score = isCorrect ? 100 : 0;
    
    setResult({
      success: isCorrect,
      message: isCorrect 
        ? `Successfully deployed to ${targetPoint}!` 
        : `Wrong Lagrange point! Target was ${targetPoint}`,
      score
    });
    
    if (isCorrect) {
      addScore(score);
      updateGameProgress('LagrangeExplorer', {
        completed: true,
        bestScore: score
      });
    }
    
    setDeploying(false);
  };
    const reset = () => {
    setDeployedSatellites([]);
    setDeploying(false);
    setResult(null);
    setError(null);
    setPhysicsData(null);
    setEducationalNotes([]);
  };
  
  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* 3D Scene */}
      <Box sx={{ flex: 1 }}>
        <Canvas camera={{ position: [0, 0, 25], fov: 60 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <OrbitControls enablePan={false} />
          
          {/* Space background */}
          <mesh>
            <sphereGeometry args={[100, 64, 64]} />
            <meshBasicMaterial color="#000033" side={THREE.BackSide} />
          </mesh>
          
          {/* Binary System */}
          <BinarySystem {...currentSystem} showOrbits={showOrbits} />
          
          {/* Lagrange Points */}
          {lagrangePoints.map((point) => (
            <LagrangePointMarker
              key={point.name}
              point={point}
              isTarget={point.name === targetPoint}
              isOccupied={deployedSatellites.some(s => 
                s.targetPoint.name === point.name && s.reached
              )}
              onSelect={(p) => !deploying && setTargetPoint(p.name)}
            />
          ))}
          
          {/* Deployed Satellites */}
          {deployedSatellites.map((satellite) => (
            <Satellite
              key={satellite.id}
              position={satellite.startPosition}
              lagrangePoint={satellite.targetPoint}
              onReachTarget={() => handleSatelliteReachTarget(satellite.id)}
            />
          ))}
          
          {/* Grid helper */}
          <gridHelper args={[40, 40]} position={[0, 0, -0.1]} />
          
          {/* Coordinate axes */}
          <axesHelper args={[5]} />
        </Canvas>
      </Box>
      
      {/* Controls */}
      <Paper sx={{ width: 400, p: 3, m: 2, maxHeight: '90vh', overflow: 'auto' }}>
        <Typography variant="h5" gutterBottom>
          🌐 Lagrange Point Explorer
        </Typography>
          <Typography variant="body2" sx={{ mb: 3 }}>
          Deploy satellites to gravitational equilibrium points (Lagrange points) 
          where they can maintain stable positions relative to two orbiting bodies.
          Real NVIDIA Modulus physics validation.
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>System</Typography>
          <ToggleButtonGroup
            value={system}
            exclusive
            onChange={(e, v) => v && setSystem(v)}
            fullWidth
            size="small"
          >
            <ToggleButton value="earth-moon">Earth-Moon</ToggleButton>
            <ToggleButton value="sun-earth">Sun-Earth</ToggleButton>
            <ToggleButton value="jupiter-io">Jupiter-Io</ToggleButton>
          </ToggleButtonGroup>
        </Box>
        
        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>Target Lagrange Point: {targetPoint}</Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {['L1', 'L2', 'L3', 'L4', 'L5'].map((point) => (
              <Chip
                key={point}
                label={point}
                onClick={() => !deploying && setTargetPoint(point)}
                color={targetPoint === point ? 'primary' : 'default'}
                variant={targetPoint === point ? 'filled' : 'outlined'}
              />
            ))}
          </Box>
        </Box>
        
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Lagrange Point Properties:
          </Typography>
          <List dense>
            <ListItem>
              <ListItemText
                primary="L1"
                secondary="Between bodies - unstable, requires station-keeping"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="L2"
                secondary="Beyond smaller body - ideal for space telescopes"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="L3"
                secondary="Opposite side - rarely used due to communication issues"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="L4 & L5"
                secondary="60° ahead/behind - stable, can trap asteroids (Trojans)"
              />
            </ListItem>
          </List>
        </Box>
          <Button
          variant="contained"
          fullWidth
          onClick={deploying ? reset : deploySatellite}
          disabled={(deploying && !result) || loading}
          sx={{ mb: 2 }}
          startIcon={loading ? <CircularProgress size={20} /> : null}
        >
          {loading ? 'Computing...' : deploying ? 'Reset' : `Deploy Satellite to ${targetPoint} (Modulus)`}
        </Button>
        
        {/* Educational Notes */}
        {educationalNotes.length > 0 && (
          <Paper sx={{ p: 2, mb: 3, bgcolor: 'info.main', color: 'info.contrastText' }}>
            <Typography variant="h6" gutterBottom>
              🎓 Lagrange Point Physics
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
            <Alert severity={result.success ? 'success' : 'error'} sx={{ mb: 2 }}>
              <Typography variant="h6">{result.message}</Typography>
              {result.success && (
                <Typography>Score: +{result.score} points</Typography>
              )}
            </Alert>
          </motion.div>
        )}
        
        <Box sx={{ mt: 3, p: 2, backgroundColor: 'background.default', borderRadius: 1 }}>
          <Typography variant="subtitle2" gutterBottom>
            Real-World Applications:
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            • James Webb Space Telescope orbits Sun-Earth L2
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            • SOHO monitors the Sun from Sun-Earth L1
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            • Jupiter has thousands of Trojan asteroids at L4 and L5
          </Typography>
          <Typography variant="body2" color="text.secondary">
            • Future space colonies might be placed at Earth-Moon L4/L5
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default LagrangeExplorerGame;