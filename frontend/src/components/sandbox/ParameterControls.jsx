import React from 'react';
import {
  Box,
  TextField,
  Typography,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import { ExpandMore, Add } from '@mui/icons-material';
import { HexColorPicker } from 'react-colorful';

const ParameterControls = ({ parameters, onChange, onAdd }) => {
  const presets = {
    Earth: {
      mass: 1,
      radius: 1,
      color: '#4169E1',
      vx: 0,
      vy: 30,
      vz: 0
    },
    Sun: {
      mass: 333000,
      radius: 10,
      color: '#FDB813',
      vx: 0,
      vy: 0,
      vz: 0
    },
    Moon: {
      mass: 0.0123,
      radius: 0.27,
      color: '#C0C0C0',
      vx: 0,
      vy: 1.022,
      vz: 0
    },
    Jupiter: {
      mass: 317.8,
      radius: 11.2,
      color: '#DAA520',
      vx: 0,
      vy: 13.1,
      vz: 0
    }
  };
  
  const applyPreset = (preset) => {
    onChange({
      ...parameters,
      ...presets[preset]
    });
  };
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Add Celestial Body
      </Typography>
      
      {/* Presets */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="body2" gutterBottom>
          Quick Presets:
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          {Object.keys(presets).map((preset) => (
            <Button
              key={preset}
              size="small"
              variant="outlined"
              onClick={() => applyPreset(preset)}
            >
              {preset}
            </Button>
          ))}
        </Box>
      </Box>
      
      {/* Basic Properties */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography>Basic Properties</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="Name"
              value={parameters.name}
              onChange={(e) => onChange({ ...parameters, name: e.target.value })}
              fullWidth
              size="small"
            />
            
            <TextField
              label="Mass"
              type="number"
              value={parameters.mass}
              onChange={(e) => onChange({ ...parameters, mass: parseFloat(e.target.value) || 0 })}
              InputProps={{
                endAdornment: <InputAdornment position="end">Earth masses</InputAdornment>
              }}
              fullWidth
              size="small"
            />
            
            <TextField
              label="Radius"
              type="number"
              value={parameters.radius}
              onChange={(e) => onChange({ ...parameters, radius: parseFloat(e.target.value) || 0 })}
              InputProps={{
                endAdornment: <InputAdornment position="end">units</InputAdornment>
              }}
              fullWidth
              size="small"
            />
            
            <Box>
              <Typography variant="body2" gutterBottom>
                Color
              </Typography>
              <HexColorPicker
                color={parameters.color}
                onChange={(color) => onChange({ ...parameters, color })}
                style={{ width: '100%', height: 150 }}
              />
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>
      
      {/* Position */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography>Initial Position</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="X Position"
              type="number"
              value={parameters.x}
              onChange={(e) => onChange({ ...parameters, x: parseFloat(e.target.value) || 0 })}
              InputProps={{
                endAdornment: <InputAdornment position="end">AU</InputAdornment>
              }}
              fullWidth
              size="small"
            />
            
            <TextField
              label="Y Position"
              type="number"
              value={parameters.y}
              onChange={(e) => onChange({ ...parameters, y: parseFloat(e.target.value) || 0 })}
              InputProps={{
                endAdornment: <InputAdornment position="end">AU</InputAdornment>
              }}
              fullWidth
              size="small"
            />
            
            <TextField
              label="Z Position"
              type="number"
              value={parameters.z}
              onChange={(e) => onChange({ ...parameters, z: parseFloat(e.target.value) || 0 })}
              InputProps={{
                endAdornment: <InputAdornment position="end">AU</InputAdornment>
              }}
              fullWidth
              size="small"
            />
          </Box>
        </AccordionDetails>
      </Accordion>
      
      {/* Velocity */}
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography>Initial Velocity</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <TextField
              label="X Velocity"
              type="number"
              value={parameters.vx}
              onChange={(e) => onChange({ ...parameters, vx: parseFloat(e.target.value) || 0 })}
              InputProps={{
                endAdornment: <InputAdornment position="end">km/s</InputAdornment>
              }}
              fullWidth
              size="small"
            />
            
            <TextField
              label="Y Velocity"
              type="number"
              value={parameters.vy}
              onChange={(e) => onChange({ ...parameters, vy: parseFloat(e.target.value) || 0 })}
              InputProps={{
                endAdornment: <InputAdornment position="end">km/s</InputAdornment>
              }}
              fullWidth
              size="small"
            />
            
            <TextField
              label="Z Velocity"
              type="number"
              value={parameters.vz}
              onChange={(e) => onChange({ ...parameters, vz: parseFloat(e.target.value) || 0 })}
              InputProps={{
                endAdornment: <InputAdornment position="end">km/s</InputAdornment>
              }}
              fullWidth
              size="small"
            />
          </Box>
        </AccordionDetails>
      </Accordion>
      
      <Button
        variant="contained"
        fullWidth
        startIcon={<Add />}
        onClick={onAdd}
        sx={{ mt: 2 }}
      >
        Add Body
      </Button>
    </Box>
  );
};

export default ParameterControls;