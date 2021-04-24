const { app, BrowserWindow, protocol } = require('electron')
const path = require('path')
const console = require('console');
if (process.env.NODE_ENV !== 'development') {
  global.__static = require('path').join(__dirname, '.').replace(/\\/g, '\\\\')
}

try {
  require('electron-reloader')(module)
} catch (_) {}

function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
      webSecurity: false,
    }
  })

  win.loadFile('index.html')
}

const input = {value: "Node.JS"};
const result = {textContent: null};

function sendToPython() {
    const python = require('child_process').spawn('python', ['./py/hello.py', input.value]);
    python.stdout.on('data', function (data) {
        const inputData = data.toString('utf8');
        console.log("Python response: ", inputData);

        result.textContent = inputData;
    });

    python.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    python.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
    });
}
sendToPython();

// And this anywhere:
function registerLocalVideoProtocol () {
  protocol.registerFileProtocol('local-video', (request, callback) => {
    const url = request.url.replace(/^local-video:\/\//, '')
    // Decode URL to prevent errors when loading filenames with UTF-8 chars or chars like "#"
    const decodedUrl = decodeURI(url) // Needed in case URL contains spaces
    try {
      // eslint-disable-next-line no-undef
      return callback(path.join(__static, decodedUrl))
    } catch (error) {
      console.error(
        'ERROR: registerLocalVideoProtocol: Could not get file path:',
        error
      )
    }
  })
}

app.on('ready', async () => {
  console.log("HI")
  registerLocalVideoProtocol()
})

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
