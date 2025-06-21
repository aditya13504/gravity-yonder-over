import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Stars, OrbitControls, Text3D, Center } from '@react-three/drei';
import { Box, Typography, Button, Container } from '@mui/material';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import * as THREE from 'three';

function SpinningPlanet() {
  const meshRef = useRef();
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.001;
    }
  });
  
  return (
    <mesh ref={meshRef} position={[3, 0, 0]}>
      <sphereGeometry args={[1.5, 64, 64]} />
      <meshStandardMaterial
        color="#4169E1"
        emissive="#4169E1"
        emissiveIntensity={0.1}
        roughness={0.5}
        metalness={0.3}
      />
    </mesh>
  );
}

function OrbitingMoon() {
  const groupRef = useRef();
  
  useFrame((state) => {
    if (groupRef.current) {
      const t = state.clock.getElapsedTime();
      groupRef.current.position.x = Math.cos(t) * 3 + 3;
      groupRef.current.position.z = Math.sin(t) * 3;
    }
  });
  
  return (
    <group ref={groupRef}>
      <mesh>
        <sphereGeometry args={[0.3, 32, 32]} />
        <meshStandardMaterial color="#C0C0C0" />
      </mesh>
    </group>
  );
}

const HeroSection = () => {
  return (
    <Box
      sx={{
        position: 'relative',
        height: '100vh',
        overflow: 'hidden',
        background: 'linear-gradient(180deg, #000428 0%, #004e92 100%)'
      }}
    >
      {/* 3D Background */}
      <Box sx={{ position: 'absolute', inset: 0 }}>
        <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
          <ambientLight intensity={0.3} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <Stars
            radius={100}
            depth={50}
            count={5000}
            factor={4}
            saturation={0}
            fade
            speed={1}
          />
          <SpinningPlanet />
          <OrbitingMoon />
          <OrbitControls
            enableZoom={false}
            enablePan={false}
            autoRotate
            autoRotateSpeed={0.5}
          />
        </Canvas>
      </Box>
      
      {/* Content Overlay */}
      <Container
        maxWidth="lg"
        sx={{
          position: 'relative',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          zIndex: 1
        }}
      >
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <Typography
            variant="h1"
            sx={{
              fontSize: { xs: '3rem', md: '5rem' },
              fontWeight: 700,
              mb: 2,
              background: 'linear-gradient(135deg, #fff 0%, #64b5f6 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent'
            }}
          >
            Gravity Yonder Over
          </Typography>
          
          <Typography
            variant="h5"
            sx={{
              mb: 4,
              color: 'rgba(255, 255, 255, 0.9)',
              fontWeight: 300
            }}
          >
            From falling apples to orbital slingshots — learn gravity the cosmic way
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                component={Link}
                to="/games"
                variant="contained"
                size="large"
                sx={{
                  px: 4,
                  py: 2,
                  fontSize: '1.2rem',
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #764ba2 0%, #667eea 100%)'
                  }
                }}
              >
                Start Playing
              </Button>
            </motion.div>
            
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button
                component={Link}
                to="/learn"
                variant="outlined"
                size="large"
                sx={{
                  px: 4,
                  py: 2,
                  fontSize: '1.2rem',
                  borderColor: 'white',
                  color: 'white',
                  '&:hover': {
                    borderColor: 'white',
                    backgroundColor: 'rgba(255, 255, 255, 0.1)'
                  }
                }}
              >
                Learn Physics
              </Button>
            </motion.div>
          </Box>
          
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1, duration: 1 }}
          >
            <Typography
              variant="body1"
              sx={{
                mt: 6,
                color: 'rgba(255, 255, 255, 0.7)'
              }}
            >
              🎮 5 Mini Games • 🔬 Physics Sandbox • 📚 Interactive Lessons • 🏆 Achievement System
            </Typography>
          </motion.div>
        </motion.div>
      </Container>
      
      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
        style={{
          position: 'absolute',
          bottom: 40,
          left: '50%',
          transform: 'translateX(-50%)'
        }}
      >
        <Box
          sx={{
            width: 30,
            height: 50,
            border: '2px solid rgba(255, 255, 255, 0.5)',
            borderRadius: 15,
            display: 'flex',
            justifyContent: 'center',
            pt: 1
          }}
        >
          <motion.div
            animate={{ y: [0, 20, 0] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          >
            <Box
              sx={{
                width: 4,
                height: 8,
                backgroundColor: 'white',
                borderRadius: 2
              }}
            />
          </motion.div>
        </Box>
      </motion.div>
    </Box>
  );
};

export default HeroSection;