import * as THREE from '/build/three.module.js';
import { OrbitControls } from '/jsm/controls/OrbitControls.js';
import Stats from '/jsm/libs/stats.module.js';
import { Lut } from './jsm/math/Lut.js';
import { Line2 } from './jsm/lines/Line2.js';
import { LineMaterial } from './jsm/lines/LineMaterial.js';
import { LineGeometry } from './jsm/lines/LineGeometry.js';
import { GeometryUtils } from './jsm/utils/GeometryUtils.js';
import { CSS2DRenderer, CSS2DObject } from './jsm/renderers/CSS2DRenderer.js';
import * as dat from '/dat/dat.gui.module.js';

document.getElementById("filepicker").addEventListener("change", function (event) {

  // let output = document.getElementById("listing");
  // for (let i=0; i<all_files.length; i++) {
  //   let item = document.createElement("li");
  //   item.innerHTML = all_files[i].webkitRelativePath;
  //   output.appendChild(item);
  // };
  // console.log("all_files = ",all_files);

  let files = Array.from(event.target.files)
    .filter(function (value, index, arr) {
      return value.name.endsWith("json") && value.name.startsWith("scopi_objects_");
    })
    .sort((a, b) => (a.name > b.name) ? 1 : ((b.name > a.name) ? -1 : 0));

  // console.log("files = ",files);

  let current_index_file_old = -1;
  let current_index_file = 0;
  let current_json_data = null;

  let dim = 0;

  let sph_nb = 0;
  let sph_positions = null;
  let sph_radius = null;
  let sph_rotations = null;
  let sph_quaternions = null;
  let sph_geometry = null;
  let sph_material = null;
  let sph_actors = null;

  let ell_nb = 0;
  let ell_positions = null;
  let ell_radius = null;
  let ell_rotations = null;
  let ell_quaternions = null;
  let ell_actors = [];

  let supell_nb = 0;
  let supell_positions = null;
  let supell_radius = null;
  let supell_rotations = null;
  let supell_quaternions = null;
  let supell_squareness = null;
  let supell_actors = [];


  let contacts_nij = null;
  let contacts_pi = null;
  let contacts_pj = null;

  update_json_data();

  let scene, camera;
  let renderer, labelRenderer;
  let controls;
  let geometry;
  let geometries = [];
  let material;
  let line, matline;
  let spotLight, ambientLight, directionalLight, PointLight;
  let parameters;

  let spheres;

  let sprite_scaling = 5.9;



  // https://observablehq.com/@d3/color-schemes
  let mycolors = ["#ffffcc", "#fffecb", "#fffec9", "#fffdc8", "#fffdc6", "#fffcc5",
    "#fffcc4", "#fffbc2", "#fffac1", "#fffac0", "#fff9be", "#fff9bd", "#fff8bb", "#fff8ba",
    "#fff7b9", "#fff6b7", "#fff6b6", "#fff5b5", "#fff5b3", "#fff4b2", "#fff4b0", "#fff3af",
    "#fff2ae", "#fff2ac", "#fff1ab", "#fff1aa", "#fff0a8", "#fff0a7", "#ffefa6", "#ffeea4",
    "#ffeea3", "#ffeda2", "#ffeda0", "#ffec9f", "#ffeb9d", "#ffeb9c", "#ffea9b", "#ffea99",
    "#ffe998", "#ffe897", "#ffe895", "#ffe794", "#ffe693", "#ffe691", "#ffe590", "#ffe48f",
    "#ffe48d", "#ffe38c", "#fee28b", "#fee289", "#fee188", "#fee087", "#fee085", "#fedf84",
    "#fede83", "#fedd82", "#fedc80", "#fedc7f", "#fedb7e", "#feda7c", "#fed97b", "#fed87a",
    "#fed778", "#fed777", "#fed676", "#fed574", "#fed473", "#fed372", "#fed270", "#fed16f",
    "#fed06e", "#fecf6c", "#fece6b", "#fecd6a", "#fecb69", "#feca67", "#fec966", "#fec865",
    "#fec764", "#fec662", "#fec561", "#fec460", "#fec25f", "#fec15e", "#fec05c", "#febf5b",
    "#febe5a", "#febd59", "#febb58", "#feba57", "#feb956", "#feb855", "#feb754", "#feb553",
    "#feb452", "#feb351", "#feb250", "#feb14f", "#feb04e", "#feae4d", "#fead4d", "#feac4c",
    "#feab4b", "#feaa4a", "#fea84a", "#fea749", "#fea648", "#fea547", "#fea347", "#fea246",
    "#fea145", "#fda045", "#fd9e44", "#fd9d44", "#fd9c43", "#fd9b42", "#fd9942", "#fd9841",
    "#fd9741", "#fd9540", "#fd9440", "#fd923f", "#fd913f", "#fd8f3e", "#fd8e3e", "#fd8d3d",
    "#fd8b3c", "#fd893c", "#fd883b", "#fd863b", "#fd853a", "#fd833a", "#fd8139", "#fd8039",
    "#fd7e38", "#fd7c38", "#fd7b37", "#fd7937", "#fd7736", "#fc7535", "#fc7335", "#fc7234",
    "#fc7034", "#fc6e33", "#fc6c33", "#fc6a32", "#fc6832", "#fb6731", "#fb6531", "#fb6330",
    "#fb6130", "#fb5f2f", "#fa5d2e", "#fa5c2e", "#fa5a2d", "#fa582d", "#f9562c", "#f9542c",
    "#f9522b", "#f8512b", "#f84f2a", "#f74d2a", "#f74b29", "#f64929", "#f64828", "#f54628",
    "#f54427", "#f44227", "#f44127", "#f33f26", "#f23d26", "#f23c25", "#f13a25", "#f03824",
    "#f03724", "#ef3524", "#ee3423", "#ed3223", "#ed3123", "#ec2f22", "#eb2e22", "#ea2c22",
    "#e92b22", "#e92921", "#e82821", "#e72621", "#e62521", "#e52420", "#e42220", "#e32120",
    "#e22020", "#e11f20", "#e01d20", "#df1c20", "#de1b20", "#dd1a20", "#dc1920", "#db1820",
    "#da1720", "#d91620", "#d81520", "#d71420", "#d51320", "#d41221", "#d31121", "#d21021",
    "#d10f21", "#cf0e21", "#ce0d21", "#cd0d22", "#cc0c22", "#ca0b22", "#c90a22", "#c80a22",
    "#c60923", "#c50823", "#c40823", "#c20723", "#c10723", "#bf0624", "#be0624", "#bc0524",
    "#bb0524", "#b90424", "#b80424", "#b60425", "#b50325", "#b30325", "#b10325", "#b00225",
    "#ae0225", "#ac0225", "#ab0225", "#a90125", "#a70126", "#a50126", "#a40126", "#a20126",
    "#a00126", "#9e0126", "#9c0026", "#9a0026", "#990026", "#970026", "#950026", "#930026",
    "#910026", "#8f0026", "#8d0026", "#8b0026", "#8a0026", "#880026", "#860026", "#840026",
    "#820026", "#800026"];


  function compute_color(v, vmin, vmax) {

    let N = mycolors.length;
    let dv = (vmax - vmin) / (N - 1);
    let pos = Math.round((v - vmin) / dv);
    let hexcolor;
    if (pos < 0) {
      hexcolor = mycolors[0];
    }
    else {
      if (pos > N - 1) {
        hexcolor = mycolors[N - 1];
      }
      else {
        hexcolor = mycolors[pos];
      }
    }
    // let r = parseInt(hexcolor.slice(1, 3), 16),
    //     g = parseInt(hexcolor.slice(3, 5), 16),
    //     b = parseInt(hexcolor.slice(5, 7), 16);
    // console.log("N = "+N+" dv = "+dv+" pos = "+pos+" hexcolor = "+hexcolor+" = ["+r+","+g+","+b+"]");
    // // return new THREE.Color( "rgb("+r+","+g+","+b+")" );
    return new THREE.Color(hexcolor);
  }

  // let geomline, matline, line;
  let lut; // lookup table
  // const stats = Stats();
  // document.body.appendChild(stats.dom);


  function init() {

    // renderer
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // scene
    scene = new THREE.Scene();
    // const background = new THREE.Color( 1, 1, 1 );
    // scene.background = background;

    // // pour tracer un cube unité
    // const cube_geometry = new THREE.BoxGeometry( 1, 1, 1 );
    // const cube_material = new THREE.MeshBasicMaterial( {color: 0x00ff00} );
    // const cube = new THREE.Mesh( cube_geometry, cube_material );
    // scene.add( cube );


    // // pour tester une forme parametree
    // var param_geometry = new THREE.ParametricGeometry( createSuperellipsoid(1, 2, 4, 1, 1), 25, 25 );
    // var param_material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
    // var param_obj = new THREE.Mesh( param_geometry, param_material );
    // console.log("param_obj = ",param_obj);
    // var quaternion = new THREE.Quaternion(1.2, 0, 2, 1).normalize(); // x,y,z,w
    // console.log("quaternion = ",quaternion);
    // param_obj.applyQuaternion(quaternion);
    // scene.add( param_obj );

    //camera
    camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 200);
    // console.log("window.innerWidth = "+window.innerWidth+" window.innerHeight = "+window.innerHeight);
    camera.position.z = 40;

    controls = new OrbitControls(camera, renderer.domElement);

    let gui = new dat.GUI();
    // parameters = {
    //   show_labels: false
    // };
    // gui.add( parameters, 'show_labels' ).name('Show labels');

    let FizzyText = function () {
      this.slider1 = 0;
    };
    let text = new FizzyText();
    let slider1 = gui.add(text, 'slider1', 0, files.length - 1);

    /* Here is the update */
    let resetSliders = function (name) {
      for (let i = 0; i < gui.__controllers.length; i++) {
        if (!gui.__controllers.property == name) {
          gui.__controllers[i].setValue(0);
        }
      }
    };
    slider1.onChange(function (value) {
      // console.log(value);
      current_index_file = Math.round(value);
      resetSliders('slider1');
    });

    gui.open();

  }

  function animate() {
    // pour rafraichir la page toutes les 2s
    // setTimeout( function() {
    //     requestAnimationFrame( animate );
    // }, 2000 );

    requestAnimationFrame(animate);

    // controls.update();
    render();
    update();
    // stats.update();
  };

  function update_json_data() {
    if (current_index_file != current_index_file_old) {
      // console.log("current index file : " + current_index_file);
      current_index_file_old = current_index_file;
      // console.log("current file : " + files[current_index_file].webkitRelativePath);
      fetch('api/json?filename=' + files[current_index_file].webkitRelativePath)
        .then(response => response.json())
        .then(
          json => {
            current_json_data = json;
          }
        ).then(
          () => {
            // console.log("current json data : ",current_json_data);
            // console.log("nb particules : ",current_json_data.objects.length);

            // nb de spheres et de superellipsoids
            sph_nb = 0;
            supell_nb = 0;
            ell_nb = 0;
            dim = current_json_data.objects[0].position.length;
            // console.log("dim = ",dim);
            current_json_data.objects.forEach(function (obj) {
              if (obj.type === "sphere") {
                sph_nb += 1;
              }
              else if (obj.type === "superellipsoid") {
                if (((dim == 2) && (obj.squareness[0] == 1)) || ((dim == 3) && (obj.squareness[0] == 1) && (obj.squareness[1] == 1))) {
                  ell_nb += 1;
                  obj.type = "ellipsoid";
                }
                else {
                  supell_nb += 1;
                }
              }
            });
            // console.log("nb spheres = "+sph_nb);
            // console.log("nb ellipsoids= "+ell_nb);
            // console.log("nb superellipsoids = "+supell_nb);

            if (sph_nb > 0) {
              sph_positions = new Float32Array(sph_nb * 3);
              sph_radius = new Float32Array(sph_nb);
              sph_rotations = new Float32Array(sph_nb * dim * dim);
              sph_quaternions = new Float32Array(sph_nb * 4);
            }
            if (ell_nb > 0) {
              ell_positions = new Float32Array(ell_nb * 3);
              ell_radius = new Float32Array(ell_nb * dim);
              ell_rotations = new Float32Array(ell_nb * dim * dim);
              ell_quaternions = new Float32Array(ell_nb * 4);
            }
            if (supell_nb > 0) {
              supell_positions = new Float32Array(supell_nb * 3);
              supell_radius = new Float32Array(supell_nb * dim);
              supell_squareness = new Float32Array(supell_nb * (dim - 1));
              supell_rotations = new Float32Array(supell_nb * dim * dim);
              supell_quaternions = new Float32Array(supell_nb * 4);
            }
            let isph = 0;
            let iell = 0;
            let isupell = 0;
            for (let i = 0; i < current_json_data.objects.length; i++) {
              if (current_json_data.objects[i].type === "sphere") {
                sph_radius[isph] = current_json_data.objects[i].radius;
                sph_positions[3 * isph] = current_json_data.objects[i].position[0];
                sph_positions[3 * isph + 1] = current_json_data.objects[i].position[1];
                if (dim == 3) {
                  sph_positions[3 * isph + 2] = current_json_data.objects[i].position[2];
                }
                else {
                  sph_positions[3 * isph + 2] = 0;
                }
                for (let k = 0; k < dim * dim; k++) {
                  sph_rotations[dim * dim * isph + k] = current_json_data.objects[i].rotation[k];
                }
                for (let k = 0; k < 4; k++) {
                  sph_quaternions[4 * isph + k] = current_json_data.objects[i].quaternion[k];
                }
                isph += 1;
              }
              else if (current_json_data.objects[i].type === "ellipsoid") {
                for (let k = 0; k < dim; k++) {
                  ell_radius[dim * iell + k] = current_json_data.objects[i].radius[k];
                }
                ell_positions[3 * iell] = current_json_data.objects[i].position[0];
                ell_positions[3 * iell + 1] = current_json_data.objects[i].position[1];
                if (dim == 3) {
                  ell_positions[3 * iell + 2] = current_json_data.objects[i].position[2];
                }
                else {
                  ell_positions[3 * iell + 2] = 0;
                }
                for (let k = 0; k < dim * dim; k++) {
                  ell_rotations[dim * dim * iell + k] = current_json_data.objects[i].rotation[k];
                }
                for (let k = 0; k < 4; k++) {
                  ell_quaternions[4 * iell + k] = current_json_data.objects[i].quaternion[k];
                }
                iell += 1;
              }
              else if (current_json_data.objects[i].type === "superellipsoid") {
                for (let k = 0; k < dim; k++) {
                  supell_radius[dim * isupell + k] = current_json_data.objects[i].radius[k];
                }
                supell_positions[3 * isupell] = current_json_data.objects[i].position[0];
                supell_positions[3 * isupell + 1] = current_json_data.objects[i].position[1];
                if (dim == 3) {
                  supell_positions[3 * isupell + 2] = current_json_data.objects[i].position[2];
                }
                else {
                  supell_positions[3 * isupell + 2] = 0;
                }
                for (let k = 0; k < dim * dim; k++) {
                  supell_rotations[dim * dim * isupell + k] = current_json_data.objects[i].rotation[k];
                }
                for (let k = 0; k < dim - 1; k++) {
                  supell_squareness[(dim - 1) * isupell + k] = current_json_data.objects[i].squareness[k];
                }
                for (let k = 0; k < 4; k++) {
                  supell_quaternions[4 * isupell + k] = current_json_data.objects[i].quaternion[k];
                }
                isupell += 1;
              }
            }
            // console.log("spheres : positions = ",sph_positions);
            // console.log("spheres : radius = ",sph_radius);
            // console.log("spheres : rotations = ",sph_rotations);
            // console.log("ellipsoids : positions = ",ell_positions);
            // console.log("ellipsoids : radius = ",ell_radius);
            // console.log("ellipsoids : rotations = ",ell_rotations);
            // console.log("superellipsoids : positions = ",supell_positions);
            // console.log("superellipsoids : radius = ",supell_radius);
            // console.log("superellipsoids : rotations = ",supell_rotations);
            // console.log("superellipsoids : squareness = ",supell_squareness);
          }
        ).then(
          () => {
            redrawParticules();
          }
        ).then(
          () => {
            render();
          }
        );
    }
  }

  function redrawParticules() {

    // SPHERES
    if (sph_nb > 0) {
      try {
        const geom = sph_actors.geometry;
        const attributes = geom.attributes;

        for (let i = 0; i < attributes.size.array.length; i++) {
          attributes.size.array[i] = sprite_scaling * sph_radius[i];
          attributes.position.array[3 * i] = sph_positions[3 * i];
          attributes.position.array[3 * i + 1] = sph_positions[3 * i + 1];
          attributes.position.array[3 * i + 2] = sph_positions[3 * i + 2];
        }
        attributes.size.needsUpdate = true;
        attributes.position.needsUpdate = true;
      } catch (error) {
        // console.error(error);
        const colors = new Float32Array(sph_nb * 3);
        const sizes = new Float32Array(sph_nb);
        const color = new THREE.Color(0xffffff);
        for (let i = 0; i < sph_nb; i++) {
          // color.setHSL( 0.5, 0.7, 0.5 );
          color.setHSL(1, 1, 1);
          color.toArray(colors, i * 3);
          sizes[i] = sprite_scaling * sph_radius[i];
        }
        const geom = new THREE.BufferGeometry();
        geom.setAttribute('position', new THREE.BufferAttribute(sph_positions, 3));
        geom.setAttribute('customColor', new THREE.BufferAttribute(colors, 3));
        geom.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        const material = new THREE.ShaderMaterial({
          uniforms: {
            color: { value: new THREE.Color(0xff0000) },
            // pointTexture: { value: new THREE.TextureLoader().load( "disc2.png" ) }
            // disc2.png : 32x32
            pointTexture: { value: new THREE.TextureLoader().load("disc_100x100.png") }
            //
            // pointTexture: { value: new THREE.TextureLoader().load( "sprite.png" ) }
          },
          vertexShader: document.getElementById('vertexshader').textContent,
          fragmentShader: document.getElementById('fragmentshader').textContent,
          blending: THREE.AdditiveBlending,
          depthTest: false,
          transparent: true
        });
        sph_actors = new THREE.Points(geom, material);
        scene.add(sph_actors);
      }
    }

    // ELLIPSOIDS
    var bar = new Promise((resolve, reject) => {
      if (ell_actors.length == 0) {
        resolve();
      }
      else {
        ell_actors.forEach((value, index, array) => {
          scene.remove(value);
          if (index === array.length - 1) resolve();
        });
      }
    });
    bar.then(() => {
      ell_actors = [];
      if (ell_nb > 0) {
        for (let i = 0; i < ell_nb; i++) {
          var rz = 0.1;
          if (dim == 3) {
            rz = ell_radius[dim * i + 2];
          }
          var param_geometry = new THREE.ParametricGeometry(createSuperellipsoid(ell_radius[dim * i], ell_radius[dim * i + 1], rz, 1, 1), 12, 12);
          var param_material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
          var param_obj = new THREE.Mesh(param_geometry, param_material);
          // console.log("param_obj = ",param_obj);
          var quaternion = new THREE.Quaternion(ell_quaternions[4 * i + 1], ell_quaternions[4 * i + 2], ell_quaternions[4 * i + 3], ell_quaternions[4 * i]).normalize(); // x,y,z,w
          // console.log("quaternion = ",quaternion);
          param_obj.applyQuaternion(quaternion);
          // var m = new THREE.Matrix4();
          // m.set( ell_quaternions[dim*dim*i],   ell_quaternions[dim*dim*i+1], 0, ell_positions[3*i],
          //        ell_quaternions[dim*dim*i+2], ell_quaternions[dim*dim*i+3], 0, ell_positions[3*i+1],
          //        0,                            0,                            1, 0,
          //        0,                            0,                            0, 1 );
          // param_obj.applyMatrix4(m);
          param_obj.position.set(ell_positions[3 * i], ell_positions[3 * i + 1], ell_positions[3 * i + 2]);
          ell_actors.push(param_obj);
          scene.add(param_obj);
        }
      }
    });
    // console.log("ell_actors = ",ell_actors);

    // SUPERELLIPSOIDS
    var bar = new Promise((resolve, reject) => {
      if (supell_actors.length == 0) {
        resolve();
      }
      else {
        supell_actors.forEach((value, index, array) => {
          scene.remove(value);
          if (index === array.length - 1) resolve();
        });
      }
    });
    bar.then(() => {
      supell_actors = [];
      if (supell_nb > 0) {
        for (let i = 0; i < supell_nb; i++) {
          var rz = 0.1;
          var sqz = 1;
          if (dim == 3) {
            rz = supell_radius[dim * i + 2];
            sqz = supell_squareness[(dim - 1) * i + 1];
          }
          var param_geometry = new THREE.ParametricGeometry(createSuperellipsoid(supell_radius[dim * i], supell_radius[dim * i + 1], rz, supell_squareness[(dim - 1) * i], sqz), 12, 12);
          var param_material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
          var param_obj = new THREE.Mesh(param_geometry, param_material);
          // console.log("param_obj = ",param_obj);
          var quaternion = new THREE.Quaternion(supell_quaternions[4 * i + 1], supell_quaternions[4 * i + 2], supell_quaternions[4 * i + 3], supell_quaternions[4 * i]).normalize(); // x,y,z,w
          // console.log("quaternion = ",quaternion);
          param_obj.applyQuaternion(quaternion);
          // var m = new THREE.Matrix4();
          // m.set( ell_quaternions[dim*dim*i],   ell_quaternions[dim*dim*i+1], 0, ell_positions[3*i],
          //        ell_quaternions[dim*dim*i+2], ell_quaternions[dim*dim*i+3], 0, ell_positions[3*i+1],
          //        0,                            0,                            1, 0,
          //        0,                            0,                            0, 1 );
          // param_obj.applyMatrix4(m);
          param_obj.position.set(supell_positions[3 * i], supell_positions[3 * i + 1], supell_positions[3 * i + 2]);
          supell_actors.push(param_obj);
          scene.add(param_obj);
        }
      }
    });
    // console.log("supell_actors = ",supell_actors);
  }

  function update() {
    // if ( keyboard.pressed("z") )
    // {
    // 	// do something
    // }
    controls.update();
    update_json_data();
    // stats.update();
  }

  function render() {
    // console.log("camera.position.z = "+camera.position.z);
    // if (camera.position.z < camera_zmin) {
    //   camera.position.z = camera_zmin;
    // }
    renderer.render(scene, camera);
    //var sss = renderer.getSize();
    //console.log("PixelRatio = "+renderer.getPixelRatio()+" DrawingBufferSize = ",sss);
  }

  init();
  animate();

  function createSuperellipsoid(rx, ry, rz, e, n) {
    return function (a, b, target) {
      a = -0.5 * Math.PI + a * Math.PI;
      b = -Math.PI + 2 * b * Math.PI;
      var cos_a = Math.cos(a);
      var sin_a = Math.sin(a);
      var cos_b = Math.cos(b);
      var sin_b = Math.sin(b);
      var x = rx * Math.sign(cos_a) * Math.pow(Math.abs(cos_a), n) * Math.sign(cos_b) * Math.pow(Math.abs(cos_b), e);
      var y = ry * Math.sign(cos_a) * Math.pow(Math.abs(cos_a), n) * Math.sign(sin_b) * Math.pow(Math.abs(sin_b), e);
      var z = rz * Math.sign(sin_a) * Math.pow(Math.abs(sin_a), n);
      target.set(x, y, z);
    }
  }



}, false);
