import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  IconButton,
  Stepper,
  Step,
  StepLabel,
  Fade,
  Backdrop
} from '@mui/material';
import {
  Close,
  NavigateBefore,
  NavigateNext,
  CheckCircle
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

const Tutorial = ({ steps, onComplete, isOpen, onClose }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [highlightElement, setHighlightElement] = useState(null);
  
  useEffect(() => {
    if (isOpen && steps[activeStep]?.element) {
      const element = document.querySelector(steps[activeStep].element);
      if (element) {
        setHighlightElement(element.getBoundingClientRect());
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [activeStep, isOpen, steps]);
  
  const handleNext = () => {
    if (activeStep < steps.length - 1) {
      setActiveStep(activeStep + 1);
    } else {
      onComplete();
      onClose();
    }
  };
  
  const handleBack = () => {
    if (activeStep > 0) {
      setActiveStep(activeStep - 1);
    }
  };
  
  const handleSkip = () => {
    onClose();
  };
  
  if (!isOpen) return null;
  
  const currentStep = steps[activeStep];
  
  return (
    <>
      <Backdrop
        sx={{
          zIndex: 9998,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          backdropFilter: 'blur(4px)'
        }}
        open={isOpen}
        onClick={handleSkip}
      />
      
      {/* Highlight Box */}
      {highlightElement && (
        <Box
          sx={{
            position: 'fixed',
            top: highlightElement.top - 10,
            left: highlightElement.left - 10,
            width: highlightElement.width + 20,
            height: highlightElement.height + 20,
            border: '3px solid',
            borderColor: 'primary.main',
            borderRadius: 2,
            zIndex: 9999,
            pointerEvents: 'none',
            boxShadow: '0 0 0 9999px rgba(0, 0, 0, 0.5)',
            animation: 'pulse 2s infinite'
          }}
        />
      )}
      
      {/* Tutorial Content */}
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
          style={{
            position: 'fixed',
            bottom: 40,
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 10000,
            maxWidth: 600,
            width: '90%'
          }}
        >
          <Paper elevation={8} sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">
                {currentStep.title}
              </Typography>
              <IconButton size="small" onClick={handleSkip}>
                <Close />
              </IconButton>
            </Box>
            
            <Typography variant="body1" sx={{ mb: 3 }}>
              {currentStep.content}
            </Typography>
            
            {currentStep.tip && (
              <Box
                sx={{
                  backgroundColor: 'info.dark',
                  borderRadius: 1,
                  p: 2,
                  mb: 3
                }}
              >
                <Typography variant="body2">
                  💡 Tip: {currentStep.tip}
                </Typography>
              </Box>
            )}
            
            <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
              {steps.map((step, index) => (
                <Step key={index}>
                  <StepLabel>{step.label || `Step ${index + 1}`}</StepLabel>
                </Step>
              ))}
            </Stepper>
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Button
                startIcon={<NavigateBefore />}
                onClick={handleBack}
                disabled={activeStep === 0}
              >
                Back
              </Button>
              
              <Button
                endIcon={activeStep === steps.length - 1 ? <CheckCircle /> : <NavigateNext />}
                variant="contained"
                onClick={handleNext}
              >
                {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
              </Button>
            </Box>
          </Paper>
        </motion.div>
      </AnimatePresence>
    </>
  );
};

export default Tutorial;