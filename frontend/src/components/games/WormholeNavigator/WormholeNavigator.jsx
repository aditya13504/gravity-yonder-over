import React, { useState, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Trail, EffectComposer, Bloom, Tunnel } from '@react-three/drei';
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
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress
} from '@mui/material';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import axios from 'axios';
import { useGameStore } from '../../../store/gameStore';
import { wormholeTraversal, CONSTANTS } from '../../../physics/einstein';

// Wormhole Component with Einstein-Rosen Bridge visualization
function Wormhole({ mass, throatRadius, stability }) {
  const meshRef = useRef();
  const tunnelRef = useRef();
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.z += 0.01;
    }
    if (tunnelRef.current) {
      tunnelRef.current.rotation.x += 0.005;
    }
  });
  
  return (
    <group>
      {/* Wormhole Throat */}
      <mesh ref={meshRef}>
        <torusGeometry args={[throatRadius / 1000, throatRadius / 3000, 16, 32]} />
        <meshStandardMaterial 
          color="#8A2BE2" 
          emissive="#4B0082" 
          emissiveIntensity={0.5}
          transparent
          opacity={0.8}
        />
      </mesh>
      
      {/* Event Horizon Rings */}
      {[0.8, 1.2, 1.6].map((scale, i) => (
        <mesh key={i} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[throatRadius * scale / 1000, throatRadius * scale * 1.1 / 1000, 32]} />
          <meshBasicMaterial 
            color="#FF1493" 
            transparent 
            opacity={0.3 - i * 0.1}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
      
      {/* Tunnel Effect */}
      <mesh ref={tunnelRef} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[throatRadius / 1200, throatRadius / 800, 20, 32, 1, true]} />
        <meshBasicMaterial 
          color="#00FFFF" 
          transparent 
          opacity={0.2}
          side={THREE.DoubleSide}
          wireframe
        />
      </mesh>
      
      {/* Exotic Matter Visualization */}
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={200}
            array={new Float32Array(Array.from({ length: 600 }, () => 
              (Math.random() - 0.5) * throatRadius / 500
            ))}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial 
          color="#FF69B4" 
          size={0.05} 
          transparent 
          opacity={0.6}
        />
      </points>
      
      {/* Stability Indicator */}
      <Text
        position={[0, throatRadius / 800, 0]}
        fontSize={0.2}
        color={stability > 0.7 ? "green" : stability > 0.4 ? "yellow" : "red"}
        anchorX="center"
        anchorY="middle"
      >
        Stability: {(stability * 100).toFixed(1)}%
      </Text>
    </group>
  );
}

// Spacecraft Component for Wormhole Navigation
function WormholeSpacecraft({ initialPosition, velocity, wormholeMass, onSuccess, onFailure, onProgress }) {
  const meshRef = useRef();
  const trailRef = useRef();
  const [trajectory, setTrajectory] = useState([]);
  const [currentPhase, setCurrentPhase] = useState('approach'); // approach, throat, exit
  
  useFrame((state, delta) => {
    if (meshRef.current) {
      const spacecraft = meshRef.current;
      const pos = spacecraft.position;
      const vel = spacecraft.userData.velocity || new THREE.Vector3(velocity, 0, 0);
      
      // Wormhole physics simulation
      const distance = pos.length();
      const throatRadius = Math.abs(wormholeMass) * 2e-27; // Scaled for visualization
      
      // Check phase
      if (distance < throatRadius * 2 && currentPhase === 'approach') {
        setCurrentPhase('throat');
        onProgress('Entering wormhole throat...');
      } else if (distance < throatRadius * 0.5 && currentPhase === 'throat') {
        setCurrentPhase('transit');
        onProgress('Transiting through spacetime...');
      } else if (distance > throatRadius * 3 && currentPhase === 'transit') {
        setCurrentPhase('exit');
        onProgress('Emerging from wormhole...');
        
        // Check if successful traversal
        const analysis = wormholeTraversal(Math.abs(wormholeMass), vel.length());
        if (analysis.isTraversable) {
          onSuccess(analysis);
        } else {
          onFailure('Spacecraft destroyed by exotic matter instabilities!');
        }
        return;
      }
      
      // Apply wormhole effects
      if (currentPhase === 'throat' || currentPhase === 'transit') {
        // Time dilation effects
        const timeDilation = 1 / Math.sqrt(1 - (vel.length() / CONSTANTS.c) ** 2);
        
        // Spacetime curvature effects
        const curvature = throatRadius / (distance + throatRadius);
        vel.multiplyScalar(1 + curvature * 0.1);
        
        // Navigate through throat
        if (currentPhase === 'throat') {
          // Funnel effect towards center
          const toCenter = new THREE.Vector3(0, 0, 0).sub(pos).normalize();
          vel.add(toCenter.multiplyScalar(0.02));
        } else if (currentPhase === 'transit') {
          // Emerge on the other side
          vel.z += 0.05; // Push through to other side
        }
      }
      
      // Update position
      pos.add(vel.clone().multiplyScalar(delta * 0.1));
      spacecraft.userData.velocity = vel;
      
      // Store trajectory for trail
      setTrajectory(prev => [...prev.slice(-50), pos.clone()]);
      
      // Check for anomalies
      if (distance > 100) {
        onFailure('Spacecraft lost in space-time anomaly!');
      }
    }
  });
  
  return (
    <group>
      <mesh ref={meshRef} position={initialPosition}>
        <coneGeometry args={[0.1, 0.3, 8]} />
        <meshStandardMaterial 
          color="#00ff00" 
          emissive="#00ff00" 
          emissiveIntensity={0.3} 
        />
      </mesh>
      
      {trajectory.length > 1 && (
        <Trail
          ref={trailRef}
          width={0.05}
          length={20}
          color={
            currentPhase === 'approach' ? new THREE.Color('#00ff00') :
            currentPhase === 'throat' ? new THREE.Color('#ff6600') :
            currentPhase === 'transit' ? new THREE.Color('#ff00ff') :
            new THREE.Color('#00ffff')
          }
          attenuation={(t) => t * t}
        >
          <mesh>
            <boxGeometry args={[0, 0, 0]} />
          </mesh>
        </Trail>
      )}
    </group>
  );
}

const WormholeNavigatorGame = () => {
  const [wormholeMass, setWormholeMass] = useState(-1e30); // Negative mass for traversable wormhole
  const [spacecraftVelocity, setSpacecraftVelocity] = useState(0.1 * CONSTANTS.c);
  const [wormholeType, setWormholeType] = useState('traversable');
  const [stabilityField, setStabilityField] = useState(true);
  const [launching, setLaunching] = useState(false);
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [physicsData, setPhysicsData] = useState(null);
  const [educationalNotes, setEducationalNotes] = useState([]);
  const { addScore, updateGameProgress } = useGameStore();
  
  const wormholeTypes = {
    traversable: { mass: -1e30, stability: 0.8, description: 'Stable Einstein-Rosen bridge' },
    unstable: { mass: -5e29, stability: 0.3, description: 'Unstable exotic matter configuration' },
    schwarzschild: { mass: 1e30, stability: 0.0, description: 'Non-traversable black hole' },
    morris_thorne: { mass: -2e30, stability: 0.9, description: 'Theoretically perfect wormhole' }
  };
  
  const currentWormhole = wormholeTypes[wormholeType];
  const throatRadius = Math.abs(currentWormhole.mass) * 2e-27;
  const stability = stabilityField ? currentWormhole.stability : currentWormhole.stability * 0.5;
  
  // Run NVIDIA Modulus wormhole navigation simulation
  const runModulusSimulation = async () => {
    setLoading(true);
    setError(null);
    setProgress('Computing wormhole traversal...');
    
    try {
      const response = await axios.post('/api/games/wormhole-navigation', {
        wormhole_mass: currentWormhole.mass,
        spacecraft_velocity: spacecraftVelocity,
        wormhole_type: wormholeType,
        stability_field: stabilityField,
        stability: stability,
        simulation_time: 100,
        time_steps: 50
      });
      
      if (response.data.success) {
        setPhysicsData(response.data.data);
        setEducationalNotes(response.data.educational_notes);
        setLaunching(true);
        addScore(response.data.score);
        setProgress('Traversal in progress...');
      } else {
        setError('Wormhole navigation simulation failed');
      }
    } catch (err) {
      setError(`Failed to run simulation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  const launchSpacecraft = () => {
    runModulusSimulation();
  };
  
  const handleSuccess = (analysis) => {
    const baseScore = 200;
    const velocityBonus = Math.min(100, (spacecraftVelocity / (0.5 * CONSTANTS.c)) * 50);
    const stabilityBonus = stability * 100;
    const score = Math.round(baseScore + velocityBonus + stabilityBonus);
    
    setResult({
      success: true,
      message: 'Wormhole traversal successful!',
      score,
      analysis: {
        timeDilation: analysis.timeDilation,
        properTime: analysis.properTime,
        stabilityUsed: stability,
        velocityFactor: spacecraftVelocity / CONSTANTS.c
      }
    });
    
    addScore(score);
    updateGameProgress('WormholeNavigator', {
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
  
  const handleProgress = (message) => {
    setProgress(message);
  };
    const reset = () => {
    setLaunching(false);
    setResult(null);
    setProgress('');
    setError(null);
    setPhysicsData(null);
    setEducationalNotes([]);
  };
  
  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)' }}>
      {/* 3D Scene */}
      <Box sx={{ flex: 1, background: 'black' }}>
        <Canvas camera={{ position: [10, 5, 10], fov: 60 }}>
          <ambientLight intensity={0.1} />
          <pointLight position={[5, 5, 5]} intensity={0.5} />
          <OrbitControls enablePan={false} />
          
          <EffectComposer>
            <Bloom luminanceThreshold={0.3} luminanceSmoothing={0.9} />
          </EffectComposer>
          
          {/* Starfield background */}
          <mesh>
            <sphereGeometry args={[100, 64, 64]} />
            <meshBasicMaterial color="#000011" side={THREE.BackSide} />
          </mesh>
          
          {/* Wormhole */}
          <Wormhole 
            mass={currentWormhole.mass} 
            throatRadius={throatRadius * 1000} 
            stability={stability}
          />
          
          {/* Spacecraft */}
          {launching && (
            <WormholeSpacecraft
              initialPosition={[8, 0, 0]}
              velocity={spacecraftVelocity}
              wormholeMass={currentWormhole.mass}
              onSuccess={handleSuccess}
              onFailure={handleFailure}
              onProgress={handleProgress}
            />
          )}
          
          {/* Coordinate grid */}
          <gridHelper args={[20, 20]} position={[0, -5, 0]} />
        </Canvas>
      </Box>
      
      {/* Controls Panel */}
      <Paper sx={{ width: 400, p: 3, m: 2, maxHeight: '90vh', overflow: 'auto' }}>        <Typography variant="h5" gutterBottom>
          🌌 Wormhole Navigator - NVIDIA Modulus
        </Typography>
        
        <Typography variant="body2" sx={{ mb: 3 }}>
          Navigate through Einstein-Rosen bridges and explore exotic spacetime geometries. 
          Master the physics of traversable wormholes using real relativistic simulations.
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <Box sx={{ mb: 3 }}>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Wormhole Type</InputLabel>
            <Select
              value={wormholeType}
              onChange={(e) => setWormholeType(e.target.value)}
              disabled={launching}
            >
              {Object.entries(wormholeTypes).map(([key, type]) => (
                <MenuItem key={key} value={key}>
                  {key.charAt(0).toUpperCase() + key.slice(1).replace('_', '-')}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {currentWormhole.description}
          </Typography>
          
          <Typography gutterBottom>
            Spacecraft Velocity: {(spacecraftVelocity / CONSTANTS.c * 100).toFixed(1)}% of c
          </Typography>
          <Slider
            value={spacecraftVelocity}
            onChange={(e, v) => setSpacecraftVelocity(v)}
            min={0.01 * CONSTANTS.c}
            max={0.9 * CONSTANTS.c}
            step={0.01 * CONSTANTS.c}
            disabled={launching}
            sx={{ mb: 2 }}
          />
          
          <FormControlLabel
            control={
              <Switch
                checked={stabilityField}
                onChange={(e) => setStabilityField(e.target.checked)}
                disabled={launching}
              />
            }
            label="Exotic Matter Stabilization Field"
          />
        </Box>
        
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Wormhole Metrics:
          </Typography>
          <Chip label={`Throat Radius: ${(throatRadius * 1000).toExponential(2)} km`} size="small" sx={{ m: 0.5 }} />
          <Chip 
            label={`Stability: ${(stability * 100).toFixed(1)}%`} 
            size="small" 
            sx={{ m: 0.5 }}
            color={stability > 0.7 ? 'success' : stability > 0.4 ? 'warning' : 'error'}
          />
          <Chip 
            label={`Mass: ${currentWormhole.mass < 0 ? 'Exotic' : 'Normal'}`} 
            size="small" 
            sx={{ m: 0.5 }}
            color={currentWormhole.mass < 0 ? 'secondary' : 'primary'}
          />
        </Box>
        
        {progress && (
          <Alert severity="info" sx={{ mb: 2 }}>
            {progress}
          </Alert>
        )}
          <Button
          variant="contained"
          fullWidth
          onClick={launching ? reset : launchSpacecraft}
          sx={{ mb: 2 }}
          disabled={loading || (!launching && (wormholeType === 'schwarzschild' && !stabilityField))}
          startIcon={loading ? <CircularProgress size={20} /> : null}
        >
          {loading ? 'Computing...' : launching ? 'Reset Mission' : 'Begin Traversal (Modulus)'}
        </Button>
        
        {/* Educational Notes */}
        {educationalNotes.length > 0 && (
          <Paper sx={{ p: 2, mb: 3, bgcolor: 'info.main', color: 'info.contrastText' }}>
            <Typography variant="h6" gutterBottom>
              🎓 Wormhole Physics
            </Typography>
            {educationalNotes.map((note, index) => (
              <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                • {note}
              </Typography>
            ))}
          </Paper>
        )}
        
        {progress && (
          <Alert severity="info" sx={{ mb: 2 }}>
            {progress}
          </Alert>
        )}
        
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Alert severity={result.success ? 'success' : 'error'} sx={{ mb: 2 }}>
              <Typography variant="h6">{result.message}</Typography>
              {result.success && (
                <>
                  <Typography>Score: +{result.score} points</Typography>
                  <Typography variant="body2">
                    Time Dilation: {result.analysis.timeDilation.toFixed(2)}x
                  </Typography>
                  <Typography variant="body2">
                    Velocity: {(result.analysis.velocityFactor * 100).toFixed(1)}% of c
                  </Typography>
                </>
              )}
            </Alert>
          </motion.div>
        )}
        
        <Box sx={{ mt: 3, p: 2, backgroundColor: 'background.default', borderRadius: 1 }}>
          <Typography variant="subtitle2" gutterBottom>
            Physics Concepts:
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            • Wormholes require exotic matter with negative energy density
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            • Traversability depends on throat stability and geometry
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            • Time dilation affects the traveler's proper time
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            • Casimir effect might provide the necessary exotic matter
          </Typography>
          
          <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
            Real Science:
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            • Morris-Thorne wormholes are theoretically possible solutions
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            • No known way to create or maintain stable wormholes exists
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            • Quantum effects likely destroy any natural wormholes
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default WormholeNavigatorGame;
