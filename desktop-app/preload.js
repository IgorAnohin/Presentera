window.addEventListener('DOMContentLoaded', () => {
  const replaceText = (selector, text) => {
    const element = document.getElementById(selector)
    if (element) element.innerText = text
  }

  for (const type of ['chrome', 'node', 'electron']) {
    replaceText(`${type}-version`, process.versions[type])
  }

  const fs = require('fs');
  const path = require('path');

  let rawdata = fs.readFileSync(path.resolve(__dirname, 'configuretion.json'));
  let videos = JSON.parse(rawdata);
  let videoLenWithOffset = videos.length + 1;
  for (var raw = 0; raw < videoLenWithOffset / 3; raw++) {
    console.log("ROW")
    $('#container' ).append(
        `
          <div id="row${raw}" class="row" style="display: flex;align-items: center;">
          </div>
        `)

    let idx = raw * 3
    for (; idx < videoLenWithOffset && idx < (raw + 1) * 3; idx ++) {
      if (idx < videos.length) {
        let video = videos[idx]
        $(`#row${raw}`).append(
          `
          <div class="col">
            <div class="dropdown">
              <button style="width: 200px;" class="btn btn-primary dropdown-toggle" type="button" id="dropdownMenuButton${idx}" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                Выберите действие${idx}
              </button>
              <div class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                <a class="dropdown-item" href="#" onclick="$('#dropdownMenuButton${idx}' ).html('Следующий слайд${idx}'); ">Следующий слайд</a>
                <a class="dropdown-item" href="#" onclick="$('#dropdownMenuButton${idx}' ).html('Предыдущий слайд${idx}'); ">Предыдущий слайд</a>
              </div>
            </div>
            <div style="height: 5px;"></div>
            <video controls autoplay width="200px" height="360px">
              <source src="local-video://IMG_4315.MOV.mp4" type="video/mp4"> -->
            </video>
          </div>
          `
        ); 
        console.log(video);
      } else {
        $(`#row${raw}`).append(
          `
          <div class="col">
            <div class="dropdown">
              <button style="width: 200px;" class="btn btn-primary dropdown-toggle" type="button" id="dropdownMenuButton${idx}" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                Выберите действие${idx}
              </button>
              <div class="dropdown-menu" aria-labelledby="dropdownMenuButton">
                <a class="dropdown-item" href="#" onclick="$('#dropdownMenuButton${idx}' ).html('Следующий слайд${idx}'); ">Следующий слайд</a>
                <a class="dropdown-item" href="#" onclick="$('#dropdownMenuButton${idx}' ).html('Предыдущий слайд${idx}'); ">Предыдущий слайд</a>
              </div>
            </div>
            <div style="height: 5px;"></div>
            <div style="display: flex;align-items: center;width: 200px;height: 365px;background-color: #D2E5FB;">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-upload" viewBox="0 0 16 16">
                <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                <path d="M7.646 1.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 2.707V11.5a.5.5 0 0 1-1 0V2.707L5.354 4.854a.5.5 0 1 1-.708-.708l3-3z"/>
              </svg>
            </div>
          </div>
          `
        ); 
      }
    }

    for (; idx < (raw + 1) * 3; idx ++) {
      $(`#row${raw}`).append(
        `
        <div class="col">
        </div>
        `
      ); 
    }
  }
})
