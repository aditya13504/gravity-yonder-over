import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { Box, Typography, LinearProgress } from '@mui/material';
import { schwarzschildRadius, timeDilation } from '../../physics/einstein';

const SpacetimeCurvature = ({ mass }) => {
  const meshRef = useRef();
  
  const geometry = useMemo(() => {
    const size = 40;
    const segments = 50;
    const geometry = new THREE.PlaneGeometry(size, size, segments, segments);
    
    const positions = geometry.attributes.position;
    const rs = schwarzschildRadius(mass * 1.989e30) / 1e9; // Convert to km then scale
    
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      const r = Math.sqrt(x * x + y * y);
      
      // Embedding diagram for Schwarzschild metric
      let z = 0;
      if (r > rs * 0.001) {
        z = -2 * Math.sqrt(rs * 0.001 * r) * Math.min(mass / 10, 2);
      }
      
      positions.setZ(i, z);
    }
    
    geometry.computeVertexNormals();
    return geometry;
  }, [mass]);
  
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.z += 0.002;
    }
  });
  
  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial
        color="#4169E1"
        wireframe
        emissive="#4169E1"
        emissiveIntensity={0.2}
      />
    </mesh>
  );
};

const TimeDilationClock = ({ mass, radius }) => {
  const [localTime, setLocalTime] = React.useState(0);
  const [dilatedTime, setDilatedTime] = React.useState(0);
  
  React.useEffect(() => {
    const interval = setInterval(() => {
      setLocalTime(prev => prev + 1);
      const dilated = timeDilation(1, mass * 1.989e30, radius * 1e9);
      setDilatedTime(prev => prev + dilated);
    }, 1000);
    
    return () => clearInterval(interval);
  }, [mass, radius]);
  
  return (
    <Html distanceFactor={10} position={[10, 5, 0]}>
      <Box sx={{ 
        backgroundColor: 'rgba(0,0,0,0.9)', 
        p: 2, 
        borderRadius: 1,
        minWidth: 200
      }}>
        <Typography variant="h6" sx={{ color: 'white' }}>
          Time Dilation
        </Typography>
        <Typography variant="body2" sx={{ color: 'white' }}>
          Observer Time: {localTime}s
        </Typography>
        <Typography variant="body2" sx={{ color: 'white' }}>
          Near BH Time: {dilatedTime.toFixed(3)}s
        </Typography>
        <Typography variant="caption" sx={{ color: 'gray' }}>
          Factor: {(dilatedTime / localTime).toFixed(5)}
        </Typography>
      </Box>
    </Html>
  );
};

const RelativisticEffectsVisualizer = ({ parameters }) => {
  const rs = schwarzschildRadius(parameters.mass * 1.989e30) / 1e9;
  const photonSphereRadius = 1.5 * rs;
  const isco = 3 * rs; // Innermost stable circular orbit
  
  return (
    <>
      <Canvas camera={{ position: [30, 20, 30], fov: 60 }}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <OrbitControls enablePan={false} />
        
        {/* Black hole */}
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[rs * 0.001, 32, 32]} />
          <meshBasicMaterial color="black" />
        </mesh>
        
        {/* Event horizon */}
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[rs * 0.001, 32, 32]} />
          <meshBasicMaterial color="red" wireframe />
        </mesh>
        
        {/* Photon sphere */}
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[photonSphereRadius * 0.001, 32, 32]} />
          <meshBasicMaterial color="yellow" wireframe opacity={0.5} transparent />
        </mesh>
        
        {/* ISCO */}
        <mesh position={[0, 0, 0]}>
          <torusGeometry args={[isco * 0.001, 0.01, 16, 100]} />
          <meshBasicMaterial color="green" />
        </mesh>
        
        {/* Spacetime curvature */}
        <SpacetimeCurvature mass={parameters.mass} />
        
        {/* Time dilation visualization */}
        <TimeDilationClock mass={parameters.mass} radius={parameters.distance} />
        
        {/* Labels */}
        <Text position={[0, -10, 0]} fontSize={0.5} color="white">
          Schwarzschild Black Hole - {parameters.mass} Solar Masses
        </Text>
      </Canvas>
      
      {/* Info panel */}
      <Box sx={{ 
        position: 'absolute', 
        bottom: 10, 
        left: 10, 
        backgroundColor: 'rgba(0,0,0,0.8)',
        p: 2,
        borderRadius: 1,
        maxWidth: 300
      }}>
        <Typography variant="h6" sx={{ color: 'white', mb: 1 }}>
          Critical Radii
        </Typography>
        <Box sx={{ mb: 1 }}>
          <Typography variant="body2" sx={{ color: 'red' }}>
            Event Horizon: {rs.toFixed(2)} km
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={100} 
            sx={{ backgroundColor: 'rgba(255,0,0,0.3)', '& .MuiLinearProgress-bar': { backgroundColor: 'red' } }}
          />
        </Box>
        <Box sx={{ mb: 1 }}>
          <Typography variant="body2" sx={{ color: 'yellow' }}>
            Photon Sphere: {photonSphereRadius.toFixed(2)} km
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={66} 
            sx={{ backgroundColor: 'rgba(255,255,0,0.3)', '& .MuiLinearProgress-bar': { backgroundColor: 'yellow' } }}
          />
        </Box>
        <Box>
          <Typography variant="body2" sx={{ color: 'green' }}>
            ISCO: {isco.toFixed(2)} km
          </Typography>
          <LinearProgress 
            variant="determinate" 
            value={33} 
            sx={{ backgroundColor: 'rgba(0,255,0,0.3)', '& .MuiLinearProgress-bar': { backgroundColor: 'green' } }}
          />
        </Box>
      </Box>
    </>
  );
};

export default RelativisticEffectsVisualizer;