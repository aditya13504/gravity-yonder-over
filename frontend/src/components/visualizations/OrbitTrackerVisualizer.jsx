import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Trail, Line, Text, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Box, Typography, Chip } from '@mui/material';

const OrbitingBody = ({ radius, velocity, color, name, showTrail }) => {
  const meshRef = useRef();
  const [trail, setTrail] = useState([]);
  const [orbitalParams, setOrbitalParams] = useState({});
  
  useFrame((state) => {
    if (meshRef.current) {
      const t = state.clock.getElapsedTime() * velocity / radius;
      meshRef.current.position.x = radius * Math.cos(t);
      meshRef.current.position.z = radius * Math.sin(t);
      
      // Update trail
      if (showTrail) {
        setTrail(prev => [...prev.slice(-100), meshRef.current.position.clone()]);
      }
      
      // Calculate orbital parameters
      if (state.clock.getElapsedTime() % 1 < 0.016) {
        const period = 2 * Math.PI * radius / velocity;
        const angularVelocity = velocity / radius;
        setOrbitalParams({ period, angularVelocity });
      }
    }
  });
  
  return (
    <>
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
        <Html distanceFactor={10}>
          <Box sx={{ 
            backgroundColor: 'rgba(0,0,0,0.8)', 
            p: 1, 
            borderRadius: 1,
            minWidth: 120
          }}>
            <Typography variant="caption" sx={{ color: 'white' }}>
              {name}
            </Typography>
            <Typography variant="caption" sx={{ color: 'white', display: 'block' }}>
              T: {orbitalParams.period?.toFixed(1)}s
            </Typography>
          </Box>
        </Html>
      </mesh>
      
      {showTrail && trail.length > 1 && (
        <Line
          points={trail}
          color={color}
          lineWidth={2}
          opacity={0.5}
          transparent
        />
      )}
    </>
  );
};

const OrbitTrackerVisualizer = ({ parameters }) => {
  const [showTrails, setShowTrails] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  
  const bodies = [
    { name: 'Inner Planet', radius: 5, velocity: parameters.velocity * 1.5, color: '#ff6b6b' },
    { name: 'Earth', radius: 10, velocity: parameters.velocity, color: '#4169E1' },
    { name: 'Outer Planet', radius: 15, velocity: parameters.velocity * 0.7, color: '#4ecdc4' }
  ];
  
  return (
    <>
      <Canvas camera={{ position: [20, 20, 20], fov: 60 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[0, 0, 0]} intensity={2} />
        <OrbitControls enablePan={false} />
        
        {/* Sun */}
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[2, 32, 32]} />
          <meshStandardMaterial 
            color="#FDB813" 
            emissive="#FDB813" 
            emissiveIntensity={0.8}
          />
        </mesh>
        
        {/* Orbiting bodies */}
        {bodies.map((body, index) => (
          <OrbitingBody
            key={index}
            {...body}
            showTrail={showTrails}
            showLabels={showLabels}
          />
        ))}
        
        {/* Orbit paths */}
        {bodies.map((body, index) => (
          <Line
            key={`orbit-${index}`}
            points={Array.from({ length: 64 }, (_, i) => {
              const angle = (i / 63) * Math.PI * 2;
              return new THREE.Vector3(
                body.radius * Math.cos(angle),
                0,
                body.radius * Math.sin(angle)
              );
            })}
            color="rgba(255,255,255,0.2)"
            lineWidth={1}
          />
        ))}
        
        {/* Reference plane */}
        <gridHelper args={[40, 40]} position={[0, -0.1, 0]} />
        
        {/* Kepler's Laws visualization */}
        <Text position={[0, -5, 0]} fontSize={0.5} color="white">
          Kepler's Third Law: T² ∝ a³
        </Text>
      </Canvas>
      
      {/* Info overlay */}
      <Box sx={{ 
        position: 'absolute', 
        top: 10, 
        right: 10, 
        backgroundColor: 'rgba(0,0,0,0.8)',
        p: 2,
        borderRadius: 1
      }}>
        <Typography variant="h6" sx={{ color: 'white', mb: 1 }}>
          Orbital Mechanics
        </Typography>
        {bodies.map((body, index) => (
          <Chip
            key={index}
            label={body.name}
            size="small"
            sx={{ 
              backgroundColor: body.color, 
              color: 'white',
              m: 0.5
            }}
          />
        ))}
      </Box>
    </>
  );
};

export default OrbitTrackerVisualizer;