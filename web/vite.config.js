import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // By default, bind strictly to localhost (single URL).
    // To share on your local Wi-Fi network, pass `--host` or change to `host: true`.
    host: 'localhost'
  }
});
