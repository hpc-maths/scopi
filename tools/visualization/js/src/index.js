import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import Stats from "three/examples/jsm/libs/stats.module";
import { GUI } from "three/examples/jsm/libs/lil-gui.module.min";
import { Lut } from "three/addons/math/Lut.js";
import { Text } from "troika-three-text";
import * as math from "mathjs";
import * as BSON from "bson";

let camera, scene, renderer, lut;
let container, controls, stats, gui, guiStatsEl;
let params;
params = {
    colorMap: "rainbow",
};

var oFiles = [];
var guiFrame;
var clock = new THREE.Clock();

var options = {
    refresh: 0.01,
    pause: true,
    contacts: true,
    radiuses: true,
    show_plan: false,
    current_frame: 0,
    use_velocity: false,
};

document
    .getElementById("uploadResult")
    .addEventListener("change", function (event) {
        oFiles = Array.from(document.getElementById("uploadResult").files).sort(
            (a, b) => (a.name > b.name ? 1 : b.name > a.name ? -1 : 0)
        );

        guiFrame.max(oFiles.length - 1);
        options.current_frame = 0;
        drawObjects();
        clock.stop();
        clock.start();
    });

init();
animate();

function createText(number, position) {
    const myText = new Text();

    // Set properties to configure:
    myText.text = number.toString();
    myText.fontSize = 0.7;
    myText.position.z = 2;
    myText.color = 0x9966ff;
    myText.position.x = position[0];
    myText.position.y = position[1];
    // console.log(myText);
    scene.add(myText);
}

/*
class ContactsCurve extends THREE.CurvePath {
    constructor(contacts) {
        super();
        contacts.forEach((obj, index) => {
            const pi = new THREE.Vector3();
            const pj = new THREE.Vector3();
            pi.x = obj.pi[0];
            pi.y = obj.pi[1];
            pj.x = obj.pj[0];
            pj.y = obj.pj[1];
            if (obj.pi.length == 2) {
                pi.z = 0.;
                pj.z = 0.;
            }
            else {
                pi.z = obj.pi[2];
                pj.z = obj.pj[2];
            }
            super.add(new THREE.LineCurve3(pi, pj));
        });
        console.log(super.curves);
    }
}
*/

const sphereObject = (function () {
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
        } else {
            // obj.radius === "object"
            scale.x = obj.radius[0];
            scale.y = obj.radius[1];
        }

        if (obj.position.length == 2) {
            position.z = 0;
            scale.z = 0.001;
        } else {
            position.z = obj.position[2];
            if (typeof obj.radius === "number") {
                scale.z = obj.radius;
            } else {
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
        if (obj.position.length == 2) {
            center.z = 0;
        } else {
            center.z = obj.position[2];
        }
        rot.push(center);
        if (typeof obj.radius === "number") {
            vec.x = obj.radius;
            vec.y = 0;
            vec.z = 0;
        } else {
            vec.x = obj.radius[0];
            vec.y = 0;
            vec.z = 0;
        }
        vec.applyQuaternion(quaternion);
        vec.add(center);

        rot.push(vec);
    };
})();

function drawObjects() {
    if (options.current_frame < oFiles.length) {
        const reader = new FileReader();
        reader.addEventListener("load", (event) => {
            const data =
                ext == "json"
                    ? JSON.parse(reader.result)
                    : BSON.deserialize(event.target.result);

            const objects = data.objects;
            const contacts = data.contacts;

            lut = new Lut();
            clean(scene);

            // const domain = new THREE.PlaneGeometry( 1, 1 );
            // const domain_material = new THREE.MeshBasicMaterial( {color: 0xffff00, tansparent: true} );
            // const domain_mesh = new THREE.Mesh( domain, domain_material );
            // domain_mesh.position.set(0.5, 0.5, 0);
            // scene.add(domain_mesh);

            var geometry = new THREE.SphereGeometry(1, 16, 16);

            var material;
            if (options.use_velocity) {
                material = new THREE.MeshPhysicalMaterial({
                    // color: "red",
                    metalness: 0.5,
                    roughness: 0.5,
                    clearcoat: 0,
                    clearcoatRoughness: 0,
                    reflectivity: 0,
                });
            } else {
                material = new THREE.MeshPhysicalMaterial({
                    color: "#0088aa",
                    metalness: 0.5,
                    roughness: 0.5,
                    clearcoat: 0,
                    clearcoatRoughness: 0,
                    reflectivity: 0,
                });
            }

            scene.add(new THREE.AmbientLight(0x222222));
            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(-1, -1, -1).normalize();
            scene.add(directionalLight);
            const particleLight = new THREE.Mesh(
                new THREE.SphereGeometry(-1, -1, -1),
                new THREE.MeshBasicMaterial({ color: 0xffffff })
            );
            scene.add(particleLight);
            const pointLight = new THREE.PointLight(0xffffff, 2, 800);
            particleLight.add(pointLight);
            // const material = new THREE.MeshBasicMaterial({ color: 'red' });

            var nbSpheres = 0;
            objects.forEach((obj, index) => {
                if (obj.type === "worm") {
                    obj.worm.forEach((sphere, indexSphere) => {
                        nbSpheres += 1;
                    });
                } else if (
                    obj.type === "sphere" ||
                    obj.type === "superellipsoid"
                ) {
                    nbSpheres += 1;
                }
            });
            var mesh = new THREE.InstancedMesh(geometry, material, nbSpheres);
            scene.add(mesh);
            const rot = [];

            const matrix = new THREE.Matrix4();
            nbSpheres = 0;

            let velocity = [];
            if (options.use_velocity) {
                lut.setColorMap(params.colorMap);

                if (objects[0].properties.velocity) {
                    objects.forEach((obj, index) => {
                        velocity.push(math.norm(obj.properties.velocity));
                    });
                }
                lut.setMin(math.min(velocity));
                lut.setMax(math.max(velocity));

                if (math.max(velocity) == math.min(velocity)) {
                    lut.setMax(math.min(velocity) + 1);
                }
            }

            objects.forEach((obj, index) => {
                if (options.show_plan && obj.type === "plane") {
                    const normal = new THREE.Vector3();
                    const position = new THREE.Vector3();
                    // TODO this is 3D only
                    normal.x = obj.normal[0];
                    normal.y = obj.normal[1];
                    position.x = obj.position[0];
                    position.y = obj.position[1];
                    if (obj.position.length == 2) {
                        normal.z = 0;
                        position.z = 0;
                    } else {
                        normal.z = obj.normal[2];
                        position.z = obj.position[2];
                    }
                    const plane = new THREE.Plane(
                        normal,
                        -position.dot(normal)
                    );
                    plane.normalize();
                    var canvasWidth = window.innerWidth;
                    const helper = new THREE.PlaneHelper(
                        plane,
                        canvasWidth,
                        "red"
                    );
                    scene.add(helper);
                } else if (obj.type === "worm") {
                    obj.worm.forEach((sphere, indexSphere) => {
                        sphereObject(sphere, matrix, rot);
                        mesh.setMatrixAt(nbSpheres, rot);
                        nbSpheres += 1;
                    });
                } else if (
                    obj.type === "sphere" ||
                    obj.type === "superellipsoid"
                ) {
                    sphereObject(obj, matrix, rot);
                    mesh.setMatrixAt(nbSpheres, matrix);
                    if (options.use_velocity && velocity) {
                        mesh.setColorAt(
                            nbSpheres,
                            lut.getColor(velocity[nbSpheres])
                        );
                    }
                    nbSpheres += 1;
                } else if (obj.type === "segment") {
                    const p1 = new THREE.Vector3();
                    const p2 = new THREE.Vector3();
                    p1.x = obj.p1[0];
                    p1.y = obj.p1[1];
                    p2.x = obj.p2[0];
                    p2.y = obj.p2[1];
                    if (obj.p1.length == 2) {
                        p1.z = 0;
                        p2.z = 0;
                    } else {
                        p1.z = obj.p1[2];
                        p2.z = obj.p2[2];
                    }
                    const points = [p1, p2];
                    const geometry = new THREE.BufferGeometry().setFromPoints(
                        points
                    );
                    const material = new THREE.LineBasicMaterial({
                        color: "red",
                    });
                    const line = new THREE.Line(geometry, material);
                    scene.add(line);
                }
            });

            if (options.use_velocity) {
                mesh.instanceColor.needsUpdate = true;
            }
            // nbSpheres = 0;
            // objects.forEach((obj, index) => {
            //     if (obj.type === "sphere") {
            //         createText(obj.id, obj.position);
            //     }
            // });
            // radius of spheres
            if (options.radiuses) {
                const line_geometry_rot =
                    new THREE.BufferGeometry().setFromPoints(rot);
                const line_material_rot = new THREE.LineBasicMaterial({
                    color: "white",
                    depthTest: false,
                });
                var line_mesh_rot = new THREE.LineSegments(
                    line_geometry_rot,
                    line_material_rot
                );
                scene.add(line_mesh_rot);
            }

            // contacts
            if (options.contacts) {
                const points = [];
                if (contacts) {
                    contacts.forEach((obj, index) => {
                        if (obj.pi.length == 2) {
                            points.push(
                                new THREE.Vector3(obj.pi[0], obj.pi[1], 0)
                            );
                            points.push(
                                new THREE.Vector3(obj.pj[0], obj.pj[1], 0)
                            );
                        } else {
                            points.push(
                                new THREE.Vector3(
                                    obj.pi[0],
                                    obj.pi[1],
                                    obj.pi[2]
                                )
                            );
                            points.push(
                                new THREE.Vector3(
                                    obj.pj[0],
                                    obj.pj[1],
                                    obj.pi[2]
                                )
                            );
                        }
                    });
                    const line_geometry =
                        new THREE.BufferGeometry().setFromPoints(points);
                    const line_material = new THREE.LineBasicMaterial({
                        color: "green",
                    });
                    var line_mesh = new THREE.LineSegments(
                        line_geometry,
                        line_material
                    );
                    scene.add(line_mesh);
                }
            }
        });

        /*
            const path = new ContactsCurve(contacts);
            const line_geometry = new THREE.TubeGeometry(path, 64, 0.01);
            const line_material = new THREE.MeshBasicMaterial( { color: 'green' } );
            const line_mesh = new THREE.Mesh( line_geometry, line_material );
            scene.add( line_mesh );
            */

        // guiStatsEl.innerHTML = [

        //     '<i>Number of objects</i>: ' + objects.length,
        //     '<i>Number of contacts</i>: ' + contacts.length

        // ].join('<br/>');

        // The X axis is red. The Y axis is green. The Z axis is blue.
        // const axesHelper = new THREE.AxesHelper( 5 );
        // scene.add( axesHelper );

        const ext = (oFiles[options.current_frame].name.match(
            /\.[0-9a-z]{1,5}$/i
        ) || [""])[0].substring(1);
        if (ext == "json") reader.readAsText(oFiles[options.current_frame]);
        else reader.readAsArrayBuffer(oFiles[options.current_frame]);
    } else {
        options.current_frame = 0;
    }
}

function clean(obj) {
    while (obj.children.length > 0) {
        clean(obj.children[0]);
        obj.remove(obj.children[0]);
    }
    if (obj.geometry) obj.geometry.dispose();

    if (obj.material) {
        //in case of map, bumpMap, normalMap, envMap ...
        Object.keys(obj.material).forEach((prop) => {
            if (!obj.material[prop]) return;
            if (
                obj.material[prop] !== null &&
                typeof obj.material[prop].dispose === "function"
            )
                obj.material[prop].dispose();
        });
        obj.material.dispose();
    }
}

function init() {
    clock.stop();
    camera = new THREE.PerspectiveCamera(
        40,
        window.innerWidth / window.innerHeight,
        0.1,
        10000
    );
    camera.position.z = 100;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputEncoding = THREE.sRGBEncoding;

    const container = document.getElementById("container");
    container.appendChild(renderer.domElement);

    stats = new Stats();
    stats.dom.style.cssText = "position:fixed;bottom:20px;left:20px";
    container.appendChild(stats.dom);

    window.addEventListener("resize", onWindowResize);

    controls = new OrbitControls(camera, renderer.domElement);

    gui = new GUI();

    var animation = gui.addFolder("Animation");
    animation.add(options, "refresh", 0.01, 5, 0.01);
    guiFrame = animation.add(options, "current_frame", 0, 0, 1).listen();
    guiFrame.onChange(function (value) {
        drawObjects();
    });
    animation.add(options, "pause").name("Pause");
    animation.open();

    var view = gui.addFolder("View");
    view.add(options, "contacts")
        .name("Draw contacts")
        .listen()
        .onChange(function (value) {
            drawObjects();
        });
    view.add(options, "radiuses")
        .name("Draw radiuses")
        .listen()
        .onChange(function (value) {
            drawObjects();
        });
    view.add(options, "show_plan")
        .name("Show plans")
        .listen()
        .onChange(function (value) {
            drawObjects();
        });
    view.open();

    var obj = {
        reset: function () {
            controls.reset();
        },
    };
    view.add(obj, "reset").name("Reset camera");

    var color_widget = gui.addFolder("Color");
    color_widget
        .add(options, "use_velocity")
        .name("Color using velocities")
        .listen()
        .onChange(function (value) {
            drawObjects();
        });

    color_widget
        .add(params, "colorMap", [
            "rainbow",
            "cooltowarm",
            "blackbody",
            "grayscale",
        ])
        .onChange(function () {
            if (options.use_velocity) {
                drawObjects();
            }
        });
    color_widget.open();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);

    if (
        !options.pause &&
        clock.running &&
        clock.getElapsedTime() > options.refresh
    ) {
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
