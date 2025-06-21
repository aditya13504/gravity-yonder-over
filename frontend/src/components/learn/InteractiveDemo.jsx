import React, { useState } from 'react';
import {
  Box,
  Typography,
  Slider,
  TextField,
  Button,
  Paper,
  Grid,
  InputAdornment,
  Divider
} from '@mui/material';
import { Calculate, Refresh } from '@mui/icons-material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import * as physics from '../../physics/newton';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const InteractiveDemo = ({ type = 'force-calculator' }) => {
  const [params, setParams] = useState({
    m1: 5.972e24, // Earth mass
    m2: 7.342e22, // Moon mass
    distance: 384400000, // Earth-Moon distance in meters
    height: 100,
    gravity: 9.81
  });
  
  const [results, setResults] = useState({});
  
  const calculate = () => {
    switch (type) {
      case 'force-calculator':
        const force = physics.gravitationalForce(params.m1, params.m2, params.distance);
        const accel1 = force / params.m1;
        const accel2 = force / params.m2;
        setResults({ force, accel1, accel2 });
        break;
        
      case 'orbital-calculator':
        const orbitalVel = physics.orbitalVelocity(params.m1, params.distance);
        const escapeVel = physics.escapeVelocity(params.m1, params.distance);
        const period = physics.orbitalPeriod(params.distance, params.m1);
        setResults({ orbitalVel, escapeVel, period });
        break;
        
      case 'free-fall':
        const time = physics.freeFallTime(params.height, params.gravity);
        const finalVelocity = physics.freeFallVelocity(params.height, params.gravity);
        const energy = physics.kineticEnergy(1, finalVelocity);
        setResults({ time, finalVelocity, energy });
        break;
        
      default:
        break;
    }
  };
  
  const generateGraphData = () => {
    const labels = [];
    const data = [];
    
    if (type === 'force-calculator') {
      // Show how force changes with distance
      for (let i = 1; i <= 10; i++) {
        const d = params.distance * i * 0.5;
        labels.push(`${(d / 1e6).toFixed(0)} Mm`);
        data.push(physics.gravitationalForce(params.m1, params.m2, d));
      }
    } else if (type === 'orbital-calculator') {
      // Show how orbital velocity changes with altitude
      for (let i = 1; i <= 10; i++) {
        const r = 6.371e6 + i * 1e6; // Earth radius + altitude
        labels.push(`${i * 1000} km`);
        data.push(physics.orbitalVelocity(params.m1, r) / 1000); // Convert to km/s
      }
    }
    
    return {
      labels,
      datasets: [
        {
          label: type === 'force-calculator' ? 'Gravitational Force (N)' : 'Orbital Velocity (km/s)',
          data,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.1
        }
      ]
    };
  };
  
  React.useEffect(() => {
    calculate();
  }, []);
  
  const renderCalculator = () => {
    switch (type) {
      case 'force-calculator':
        return (
          <>
            <Typography variant="h6" gutterBottom>
              Gravitational Force Calculator
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Mass 1"
                  type="number"
                  value={params.m1}
                  onChange={(e) => setParams({ ...params, m1: parseFloat(e.target.value) })}
                  InputProps={{
                    endAdornment: <InputAdornment position="end">kg</InputAdornment>
                  }}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Mass 2"
                  type="number"
                  value={params.m2}
                  onChange={(e) => setParams({ ...params, m2: parseFloat(e.target.value) })}
                  InputProps={{
                    endAdornment: <InputAdornment position="end">kg</InputAdornment>
                  }}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12}>
                <Typography gutterBottom>
                  Distance: {(params.distance / 1000).toFixed(0)} km
                </Typography>
                <Slider
                  value={params.distance}
                  onChange={(e, v) => setParams({ ...params, distance: v })}
                  min={1000000}
                  max={1000000000}
                  step={1000000}
                />
              </Grid>
            </Grid>
            
            <Button
              variant="contained"
              startIcon={<Calculate />}
              onClick={calculate}
              fullWidth
              sx={{ my: 2 }}
            >
              Calculate
            </Button>
            
            {results.force !== undefined && (
              <Paper sx={{ p: 2, backgroundColor: 'primary.dark' }}>
                <Typography variant="h6">Results:</Typography>
                <Typography>Force: {results.force.toExponential(3)} N</Typography>
                <Typography>Acceleration on m₁: {results.accel1.toExponential(3)} m/s²</Typography>
                <Typography>Acceleration on m₂: {results.accel2.toExponential(3)} m/s²</Typography>
              </Paper>
            )}
          </>
        );
        
      case 'orbital-calculator':
        return (
          <>
            <Typography variant="h6" gutterBottom>
              Orbital Mechanics Calculator
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  label="Central Body Mass"
                  type="number"
                  value={params.m1}
                  onChange={(e) => setParams({ ...params, m1: parseFloat(e.target.value) })}
                  InputProps={{
                    endAdornment: <InputAdornment position="end">kg</InputAdornment>
                  }}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12}>
                <Typography gutterBottom>
                  Orbital Radius: {(params.distance / 1000).toFixed(0)} km
                </Typography>
                <Slider
                  value={params.distance}
                  onChange={(e, v) => setParams({ ...params, distance: v })}
                  min={6371000} // Earth radius
                  max={400000000} // Beyond Moon
                  step={1000000}
                />
              </Grid>
            </Grid>
            
            <Button
              variant="contained"
              startIcon={<Calculate />}
              onClick={calculate}
              fullWidth
              sx={{ my: 2 }}
            >
              Calculate
            </Button>
            
            {results.orbitalVel !== undefined && (
              <Paper sx={{ p: 2, backgroundColor: 'primary.dark' }}>
                <Typography variant="h6">Results:</Typography>
                <Typography>Orbital Velocity: {(results.orbitalVel / 1000).toFixed(2)} km/s</Typography>
                <Typography>Escape Velocity: {(results.escapeVel / 1000).toFixed(2)} km/s</Typography>
                <Typography>Orbital Period: {(results.period / 3600).toFixed(2)} hours</Typography>
              </Paper>
            )}
          </>
        );
        
      default:
        return null;
    }
  };
  
  return (
    <Box>
      {renderCalculator()}
      
      <Divider sx={{ my: 3 }} />
      
      <Typography variant="h6" gutterBottom>
        Visualization
      </Typography>
      <Box sx={{ height: 300 }}>
        <Line
          data={generateGraphData()}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: 'top',
              },
              title: {
                display: true,
                text: type === 'force-calculator' 
                  ? 'Force vs Distance' 
                  : 'Orbital Velocity vs Altitude'
              }
            }
          }}
        />
      </Box>
    </Box>
  );
};

export default InteractiveDemo;