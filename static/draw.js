var drawing = false;

var context;

var offset_left = 0;
var offset_top = 0;

function start_canvas() {
  var scribbler = document.getElementById("the_stage");
  context = scribbler.getContext("2d");
  scribbler.onmousedown = function (event) {
    mousedown(event);
  };
  scribbler.onmousemove = function (event) {
    mousemove(event);
  };
  scribbler.onmouseup = function (event) {
    mouseup(event);
  };
  for (var o = scribbler; o; o = o.offsetParent) {
    offset_left += o.offsetLeft - o.scrollLeft;
    offset_top += o.offsetTop - o.scrollTop;
  }
  draw();
}

function getPosition(evt) {
  evt = evt ? evt : event ? event : null;
  var left = 0;
  var top = 0;
  var scribbler = document.getElementById("the_stage");

  if (evt.pageX) {
    left = evt.pageX;
    top = evt.pageY;
  } else if (document.documentElement.scrollLeft) {
    left = evt.clientX + document.documentElement.scrollLeft;
    top = evt.clientY + document.documentElement.scrollTop;
  } else {
    left = evt.clientX + document.body.scrollLeft;
    top = evt.clientY + document.body.scrollTop;
  }
  left -= offset_left;
  top -= offset_top;

  return { x: left, y: top };
}

function mousedown(event) {
  drawing = true;
  const location = getPosition(event);
  context.lineWidth = 8.0;
  context.strokeStyle = "#000000";
  context.beginPath();
  context.moveTo(location.x, location.y);
}

function mousemove(event) {
  if (!drawing) return;
  const location = getPosition(event);
  context.lineTo(location.x, location.y);
  context.stroke();
}

function mouseup(event) {
  if (!drawing) return;
  mousemove(event);
  drawing = false;
}

function draw() {
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, 300, 300);
}

function isCanvasBlank(canvas) {
  const context = canvas.getContext("2d");

  const pixelBuffer = new Uint32Array(
    context.getImageData(0, 0, canvas.width, canvas.height).data.buffer
  );

  return !pixelBuffer.some((color) => color !== 0);
}

function clearCanvas() {
  context.clearRect(0, 0, 300, 300);

  draw();
  document.getElementById("result_box").innerHTML = " ";
}

function train() {
  const button = document.querySelector(".bsecond");
  const url = button.name;

  const canvas = document.getElementById("the_stage");
  const image = canvas.toDataURL();

  fetch(url, {
    method: "POST",
    body: image,
  })
    .then((response) => response.text())
    .then((data) => {
      alert(data);
    })
    .catch((error) => console.error("We'v got an error"));

  return;
}

function predict() {
  const button = document.querySelector(".bthird");
  const url = button.name;

  const canvas = document.getElementById("the_stage");
  const result = document.getElementById("result_box");
  const image = canvas.toDataURL();

  fetch(url, {
    method: "POST",
    body: image,
  })
    .then((response) => response.text())
    .then((data) => {
      result.innerHTML = data;
    })
    .catch((error) => alert($`We'v got an error | {error}`));

  return;
}

onload = start_canvas;
