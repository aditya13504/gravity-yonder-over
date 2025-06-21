import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Sphere } from '@react-three/drei';
import * as THREE from 'three';
import { Box, Typography, Slider } from '@mui/material';

const TidallyDistortedBody = ({ primaryMass, distance, bodyRadius }) => {
  const meshRef = useRef();
  const originalGeometry = useRef();
  
  const geometry = useMemo(() => {
    const geo = new THREE.SphereGeometry(bodyRadius, 32, 32);
    originalGeometry.current = geo.clone();
    return geo;
  }, [bodyRadius]);
  
  useFrame(() => {
    if (meshRef.current && originalGeometry.current) {
      const positions = meshRef.current.geometry.attributes.position;
      const originalPositions = originalGeometry.current.attributes.position;
      
      // Calculate tidal distortion
      const tidalStrength = (primaryMass / Math.pow(distance, 3)) * 0.001;
      
      for (let i = 0; i < positions.count; i++) {
        const x = originalPositions.getX(i);
        const y = originalPositions.getY(i);
        const z = originalPositions.getZ(i);
        
        // Apply tidal stretching along x-axis (towards primary)
        const stretchFactor = 1 + tidalStrength * x * 2;
        const squeezeFactor = 1 - tidalStrength * Math.sqrt(y * y + z * z) * 0.5;
        
        positions.setX(i, x * stretchFactor);
        positions.setY(i, y * squeezeFactor);
        positions.setZ(i, z * squeezeFactor);
      }
      
      positions.needsUpdate = true;
      meshRef.current.geometry.computeVertexNormals();
    }
  });
  
  return (
    <mesh ref={meshRef} geometry={geometry} position={[distance, 0, 0]}>
      <meshStandardMaterial 
        color="#4169E1" 
        wireframe={false}
        emissive="#4169E1"
        emissiveIntensity={0.1}
      />
    </mesh>
  );
};

const TidalForceArrows = ({ primaryMass, distance, bodyRadius }) => {
  const arrows = useMemo(() => {
    const arrowData = [];
    const tidalStrength = (primaryMass / Math.pow(distance, 3)) * bodyRadius;
    
    // Create arrows showing tidal forces
    const positions = [
      { pos: [distance + bodyRadius, 0, 0], dir: [1, 0, 0], strength: tidalStrength },
      { pos: [distance - bodyRadius, 0, 0], dir: [-1, 0, 0], strength: tidalStrength },
      { pos: [distance, bodyRadius, 0], dir: [0, -0.5, 0], strength: tidalStrength * 0.5 },
      { pos: [distance, -bodyRadius, 0], dir: [0, 0.5, 0], strength: tidalStrength * 0.5 },
      { pos: [distance, 0, bodyRadius], dir: [0, 0, -0.5], strength: tidalStrength * 0.5 },
      { pos: [distance, 0, -bodyRadius], dir: [0, 0, 0.5], strength: tidalStrength * 0.5 }
    ];
    
    positions.forEach(({ pos, dir, strength }) => {
      const origin = new THREE.Vector3(...pos);
      const direction = new THREE.Vector3(...dir).normalize();
      const length = Math.min(strength * 0.1, 2);
      arrowData.push({ origin, direction, length });
    });
    
    return arrowData;
  }, [primaryMass, distance, bodyRadius]);
  
  return (
    <>
      {arrows.map((arrow, index) => (
        <arrowHelper
          key={index}
          args={[arrow.direction, arrow.origin, arrow.length, 0xff0000, 0.5, 0.2]}
        />
      ))}
    </>
  );
};

const RocheLimit = ({ primaryRadius, primaryMass, satelliteDensity }) => {
  const fluidLimit = 2.456 * primaryRadius * Math.pow(1, 1/3); // Simplified
  const rigidLimit = 2.455 * primaryRadius * Math.pow(1, 1/3);
  
  return (
    <>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[fluidLimit, 0.05, 16, 100]} />
        <meshBasicMaterial color="red" opacity={0.5} transparent />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[rigidLimit, 0.05, 16, 100]} />
        <meshBasicMaterial color="orange" opacity={0.5} transparent />
      </mesh>
    </>
  );
};

const TidalForcesVisualizer = ({ parameters }) => {
  const primaryRadius = 2;
  
  return (
    <>
      <Canvas camera={{ position: [20, 10, 20], fov: 60 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[0, 0, 0]} intensity={2} />
        <OrbitControls enablePan={false} />
        
        {/* Primary body (e.g., planet) */}
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[primaryRadius, 64, 64]} />
          <meshStandardMaterial 
            color="#DAA520" 
            emissive="#DAA520" 
            emissiveIntensity={0.3}
          />
        </mesh>
        
        {/* Tidally distorted satellite */}
        <TidallyDistortedBody
          primaryMass={parameters.mass}
          distance={parameters.distance}
          bodyRadius={1}
        />
        
        {/* Tidal force arrows */}
        <TidalForceArrows
          primaryMass={parameters.mass}
          distance={parameters.distance}
          bodyRadius={1}
        />
        
        {/* Roche limit indicators */}
        <RocheLimit
          primaryRadius={primaryRadius}
          primaryMass={parameters.mass}
          satelliteDensity={1}
        />
        
        {/* Grid helper */}
        <gridHelper args={[30, 30]} position={[0, -5, 0]} />
        
        {/* Labels */}
        <Text position={[0, -7, 0]} fontSize={0.5} color="white">
          Tidal Forces and Roche Limit
        </Text>
      </Canvas>
      
      {/* Info panel */}
      <Box sx={{ 
        position: 'absolute', 
        top: 10, 
        left: 10, 
        backgroundColor: 'rgba(0,0,0,0.8)',
        p: 2,
        borderRadius: 1,
        maxWidth: 300
      }}>
        <Typography variant="h6" sx={{ color: 'white', mb: 1 }}>
          Tidal Effects
        </Typography>
        <Typography variant="body2" sx={{ color: 'white', mb: 1 }}>
          Tidal forces arise from the gradient in gravitational field across an extended body.
        </Typography>
        <Typography variant="caption" sx={{ color: 'red' }}>
          Red circle: Fluid Roche limit
        </Typography>
        <br />
        <Typography variant="caption" sx={{ color: 'orange' }}>
          Orange circle: Rigid body Roche limit
        </Typography>
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" sx={{ color: 'white' }}>
            Examples:
          </Typography>
          <Typography variant="caption" sx={{ color: 'gray' }}>
            • Earth-Moon: Causes ocean tides<br />
            • Jupiter-Io: Volcanic activity<br />
            • Saturn's rings: Within Roche limit
          </Typography>
        </Box>
      </Box>
    </>
  );
};

export default TidalForcesVisualizer;