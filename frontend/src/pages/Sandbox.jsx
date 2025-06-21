import React from 'react';
import { Box } from '@mui/material';
import GravityPlayground from '../components/sandbox/GravityPlayground';

const Sandbox = () => {
  return (
    <Box sx={{ height: 'calc(100vh - 64px)' }}>
      <GravityPlayground />
    </Box>
  );
};

export default Sandbox;