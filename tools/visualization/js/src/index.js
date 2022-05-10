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
    const position = new THREE.Vector3();
    const rotation = new THREE.Euler();
    const scale = new THREE.Vector3();
    return function (obj, matrix, quaternion) {
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
            scale.z = obj.radius[2];
        }
        quaternion.x = obj.quaternion[1];
        quaternion.y = obj.quaternion[2];
        quaternion.z = obj.quaternion[3];
        quaternion.w = obj.quaternion[0];
        quaternion.normalize();

        matrix.compose(position, quaternion, scale);
    };
}();

const planObject = function() {
    let xB = 0.;
    let yB = 0.;
    return function(obj, plan) {
        const c = obj.normal[0] * obj.position[0] + obj.normal[1] * obj.position[1];
        // plan.push(new THREE.Vector3(obj.position[0], obj.position[0], 0.));
        if (obj.normal[1] === 0.) { // vertical straight line 
            xB = c/obj.normal[0];
            yB = obj.position[0] + 30;
        }
        else {
            xB = 30.;
            yB = (c - obj.normal[0] * xB) / obj.normal[1];
        }
        // plan.push(new THREE.Vector3(xB, yB, 0.));



        var point1 = {
            x : obj.position[0],
            y : obj.position[0],
        };

        var point2 = {
            x : xB,
            y : yB,
        };

        var a = (point2.y - point1.y) / (point2.x - point1.x);
        var b = (point1.y * point2.x - point2.y * point1.x) / (point2.x - point1.x);

        var canvasWidth = window.innerWidth;
        var canvasHeight = window.innerHeight;

        var leftSideY = b;
        var rightSideY = (canvasWidth * a) + b;
        var topSideX = (-b) / a;
        var bottomSideX = (canvasHeight - b) / a;

        // vertical line
        if ([Infinity, -Infinity].includes(a)) {
            topSideX = bottomSideX = point1.x;
        }
        // same points
        if (a !== a) {
            throw new Error("point1 and point2 are the same")
        }

        console.log(leftSideY);

        const edgePoints = [
            {x: 0, y: leftSideY},
            {x: canvasWidth, y: rightSideY},
            {x: topSideX, y: 0},
            {x: bottomSideX, y: canvasHeight}
        ].filter(({x, y}) => x >= 0 && x <= canvasWidth && y >= 0 && y <= canvasHeight);

        plan.push(new THREE.Vector3(edgePoints[0].x, edgePoints[0].y, 0));
        plan.push(new THREE.Vector3(edgePoints[1].x, edgePoints[1].y, 0));

// context.moveTo(edgePoints[0].x , edgePoints[0].y || 0);
// context.lineTo(edgePoints[1].x || point2.x, edgePoints[1].y || canvasHeight);
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
            var mesh = new THREE.InstancedMesh(geometry, material, objects.length);
            scene.add(mesh);
            const plan = [];
            const rot = []

            const matrix = new THREE.Matrix4();
            const quaternion = new THREE.Quaternion();
            let center = new THREE.Vector3();
            let vec = new THREE.Vector3();

            objects.forEach((obj, index) => {
                if (obj.type === "plan") {
                    planObject(obj, plan);
                }
                else {
                    sphereObject(obj, matrix, quaternion);
                    mesh.setMatrixAt(index, matrix);

                    center = new THREE.Vector3(obj.position[0], obj.position[1], 0.);
                    rot.push(center);
                    if (typeof obj.radius === "number") {
                        vec = new THREE.Vector3(obj.radius, 0., 0.);
                    }
                    else {
                        vec = new THREE.Vector3(obj.radius[0], 0., 0.);
                    }
                    vec.applyQuaternion(quaternion);
                    vec.add(center);
                    rot.push(vec);
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
