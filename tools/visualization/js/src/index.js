import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import Stats from 'three/examples/jsm/libs/stats.module'
import { GUI } from 'three/examples/jsm/libs/dat.gui.module'

let camera, scene, renderer;
let container, controls, stats, gui, guiStatsEl;
var oFiles = [];
var guiFrame;
var clock = new THREE.Clock();

var options = {
    "refresh": 0.2,
    "pause": true,
    "current_frame": 0,
};

document.getElementById("uploadResult").addEventListener("change", function (event) {

    oFiles = Array.from(document.getElementById("uploadResult").files)
        .sort((a, b) => (a.name > b.name) ? 1 : ((b.name > a.name) ? -1 : 0));

    guiFrame.max(oFiles.length - 1);
    options.current_frame = 0;
    drawObjects();
    clock.stop();
    clock.start();
});

init();
animate();

const sphereObject = function () {
    return function (obj, matrix, rot) {
        const position = new THREE.Vector3();
        const quaternion = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        const center = new THREE.Vector3();
        const vec = new THREE.Vector3();

        position.x = obj.position[0];
        position.y = obj.position[1];

        if (typeof obj.radius === "number") {
            scale.x = obj.radius;
            scale.y = obj.radius;
        }
        else { // obj.radius === "object"
            scale.x = obj.radius[0];
            scale.y = obj.radius[1];
        }

        if (obj.position.length == 2) {
            position.z = 0;
            scale.z = 0.001;
        }
        else {
            position.z = obj.position[2];
            if (typeof obj.radius === "number") {
                scale.z = obj.radius;
            }
            else {
                scale.z = obj.radius[2];
            }
        }
        quaternion.x = obj.quaternion[1];
        quaternion.y = obj.quaternion[2];
        quaternion.z = obj.quaternion[3];
        quaternion.w = obj.quaternion[0];
        quaternion.normalize();

        matrix.compose(position, quaternion, scale);

        center.x = obj.position[0];
        center.y = obj.position[1];
        center.z = 0.;
        rot.push(center);
        if (typeof obj.radius === "number") {
            vec.x = obj.radius;
            vec.y = 0.;
            vec.z = 0.;
        }
        else {
            vec.x = obj.radius[0];
            vec.y = 0.;
            vec.z = 0.;
        }
        vec.applyQuaternion(quaternion);
        vec.add(center);
        rot.push(vec);
    };
}();

const planObject = function() {
    return function(obj, plan) {

        var canvasWidth = window.innerWidth;
        var canvasHeight = window.innerHeight;

        if (Math.abs(obj.normal[1]) < 1e-12) { // vertical straight line 
            plan.push(new THREE.Vector3(obj.position[0], -canvasHeight, 0));
            plan.push(new THREE.Vector3(obj.position[0],  canvasHeight, 0));
        }
        else if (Math.abs(obj.normal[0]) < 1e-12) { // horizontal straight line 
            plan.push(new THREE.Vector3(-canvasWidth, obj.position[1], 0));
            plan.push(new THREE.Vector3( canvasWidth, obj.position[1], 0));
        }
        else {
            var a = -obj.normal[0]/obj.normal[1];
            var b = (obj.normal[0] * obj.position[0] + obj.normal[1] * obj.position[1]) / obj.normal[1];

            plan.push(new THREE.Vector3((-canvasHeight - b)/a, -canvasHeight, 0));
            plan.push(new THREE.Vector3(( canvasHeight - b)/a,  canvasHeight, 0));
        }
    };
}();

function drawObjects() {

    if (options.current_frame < oFiles.length) {
        const reader = new FileReader();
        reader.addEventListener('load', (event) => {
            const data = JSON.parse(reader.result);
            const objects = data.objects;
            const contacts = data.contacts;

            clean(scene);

            var geometry = new THREE.SphereGeometry(1, 16, 16);
            const material = new THREE.MeshBasicMaterial({ color: 'red' });

            var nbSpheres = 0;
            objects.forEach((obj, index) => {
                if (obj.type === "globule") {
                    nbSpheres += 6;
                }
                else if (obj.type === "sphere" || obj.type === "superellipsoid") {
                    nbSpheres += 1;
                }
            });
            var mesh = new THREE.InstancedMesh(geometry, material, nbSpheres);
            scene.add(mesh);
            const plan = [];
            const rot = [];

            const matrix = new THREE.Matrix4();
            nbSpheres = 0;

            objects.forEach((obj, index) => {
                if (obj.type === "plan") {
                    planObject(obj, plan);
                }
                else if (obj.type === "globule") {
                    obj.globule.forEach((sphere, indexSphere) => {
                        sphereObject(sphere, matrix, rot);
                        mesh.setMatrixAt(nbSpheres, matrix);
                        nbSpheres += 1;
                    });
                }
                else {
                    sphereObject(obj, matrix, rot);
                    mesh.setMatrixAt(nbSpheres, matrix);
                    nbSpheres += 1;
                }

            });
            const line_geometry_plan = new THREE.BufferGeometry().setFromPoints(plan);
            const line_material_plan = new THREE.LineBasicMaterial({
                color: 'red',
            });
            var line_mesh_plan = new THREE.LineSegments(line_geometry_plan, line_material_plan);
            scene.add(line_mesh_plan);

            const line_geometry_rot = new THREE.BufferGeometry().setFromPoints(rot);
            const line_material_rot = new THREE.LineBasicMaterial({
                color: 'blue',
                depthTest: false
            });
            var line_mesh_rot = new THREE.LineSegments(line_geometry_rot, line_material_rot);
            scene.add(line_mesh_rot);

            const points = [];
            contacts.forEach((obj, index) => {
                if (obj.pi.length == 2) {
                    points.push(new THREE.Vector3(obj.pi[0], obj.pi[1], 0.));
                    points.push(new THREE.Vector3(obj.pj[0], obj.pj[1], 0.));
                }
                else {
                    points.push(new THREE.Vector3(obj.pi[0], obj.pi[1], obj.pi[2]));
                    points.push(new THREE.Vector3(obj.pj[0], obj.pj[1], obj.pi[2]));
                }
            });
            const line_geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line_material = new THREE.LineBasicMaterial({
                color: 'green',
            });
            var line_mesh = new THREE.LineSegments(line_geometry, line_material);
            scene.add(line_mesh);

            guiStatsEl.innerHTML = [

                '<i>Number of objects</i>: ' + objects.length,
                '<i>Number of contacts</i>: ' + contacts.length

            ].join('<br/>');

        });
        reader.readAsText(oFiles[options.current_frame]);
    }
    else {
        options.current_frame = 0;
    }
};

function clean(obj) {
    while (obj.children.length > 0) {
        clean(obj.children[0])
        obj.remove(obj.children[0]);
    }
    if (obj.geometry) obj.geometry.dispose()

    if (obj.material) {
        //in case of map, bumpMap, normalMap, envMap ...
        Object.keys(obj.material).forEach(prop => {
            if (!obj.material[prop])
                return
            if (obj.material[prop] !== null && typeof obj.material[prop].dispose === 'function')
                obj.material[prop].dispose()
        })
        obj.material.dispose()
    }
}

function init() {
    clock.stop();
    camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 10000);
    camera.position.z = 100;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputEncoding = THREE.sRGBEncoding;

    const container = document.getElementById('container');
    container.appendChild(renderer.domElement);

    stats = new Stats();
    container.appendChild(stats.dom);

    window.addEventListener('resize', onWindowResize);

    controls = new OrbitControls(camera, renderer.domElement);

    gui = new GUI();
    gui.add(options, 'refresh', 0.01, 5, 0.01);
    guiFrame = gui.add(options, 'current_frame', 0, 0, 1).listen();
    guiFrame.onChange(
        function (value) {
            drawObjects();
        });

    gui.add(options, 'pause').name('Pause');

    var obj = {
        reset: function () {
            controls.reset();
        }
    }
    gui.add(obj, 'reset').name("Reset camera");

    const infoFolder = gui.addFolder('Informations');

    guiStatsEl = document.createElement('li');
    guiStatsEl.classList.add('gui-stats');

    infoFolder.__ul.appendChild(guiStatsEl);
    infoFolder.open();
}

function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);

}

function animate() {
    requestAnimationFrame(animate);

    if (!options.pause && clock.running && clock.getElapsedTime() > options.refresh) {
        clock.stop();
        clock.start();
        options.current_frame++;
        drawObjects();
    }

    stats.update();
    controls.update();
    render();
}

function render() {
    renderer.render(scene, camera);
}
