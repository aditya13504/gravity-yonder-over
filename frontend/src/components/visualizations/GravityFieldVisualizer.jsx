import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

const GravityFieldMesh = ({ mass, distance }) => {
  const meshRef = useRef();
  
  const { positions, colors } = useMemo(() => {
    const size = 50;
    const positions = [];
    const colors = [];
    
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const x = (i - size / 2) * 0.5;
        const y = (j - size / 2) * 0.5;
        
        // Calculate gravitational potential
        const r = Math.sqrt(x * x + y * y);
        const potential = r > 0.1 ? -mass / r : -mass / 0.1;
        const z = potential * 0.5;
        
        positions.push(x, y, z);
        
        // Color based on field strength
        const strength = Math.abs(potential);
        const hue = 240 - Math.min(strength * 20, 240);
        const color = new THREE.Color().setHSL(hue / 360, 1, 0.5);
        colors.push(color.r, color.g, color.b);
      }
    }
    
    return {
      positions: new Float32Array(positions),
      colors: new Float32Array(colors)
    };
  }, [mass]);
  
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.z += 0.001;
    }
  });
  
  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.1} vertexColors />
    </points>
  );
};

const GravityFieldVisualizer = ({ parameters }) => {
  return (
    <Canvas camera={{ position: [15, 15, 15], fov: 60 }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <OrbitControls enablePan={false} />
      
      {/* Central mass */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[1, 32, 32]} />
        <meshStandardMaterial 
          color="#FDB813" 
          emissive="#FDB813" 
          emissiveIntensity={0.5}
        />
      </mesh>
      
      {/* Gravity field visualization */}
      <GravityFieldMesh mass={parameters.mass} distance={parameters.distance} />
      
      {/* Grid helper */}
      <gridHelper args={[30, 30]} position={[0, -10, 0]} />
      
      {/* Labels */}
      <Text position={[0, -12, 0]} fontSize={0.5} color="white">
        Gravitational Potential Field
      </Text>
      
      <Text position={[0, 5, 0]} fontSize={0.3} color="white">
        Mass: {parameters.mass} Solar Masses
      </Text>
    </Canvas>
  );
};

export default GravityFieldVisualizer;