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

  let rawdata = fs.readFileSync(path.resolve(__dirname, 'templates.json'));
  let videos = JSON.parse(rawdata);
  let videoLenWithOffset = videos.length;
  for (var raw = 0; raw < videoLenWithOffset / 3; raw++) {
    console.log("ROW")
    $('#container' ).append(
        `
          <div id="row${raw}" class="row" style="display: flex;align-items: center;">
          </div>
        `)

    let idx = raw * 3
    for (; idx < videoLenWithOffset && idx < (raw + 1) * 3; idx ++) {
      let video = videos[idx]
      $(`#row${raw}`).append(
        `
        <div class="col">
          <button onclick="changeAction(${idx}, ${video.src}); " style="width: 200px;" class="btn btn-primary" type="button" id="dropdownMenuButton${idx}" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
            "Выбрать"
          </button>
          <div style="height: 5px;"></div>
          <video controls autoplay width="200px" height="360px">
            <source src="local-video://${video.src}" type="video/mp4"> -->
          </video>
        </div>
        `
      ); 
      console.log(video);
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
