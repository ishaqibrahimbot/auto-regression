//Two arrays/lists that will store the user input in the form of mouseX, mouseY
let userX = [];
let userY = [];

//The coefficients of the polynomial that must be trained
let e, f, g, h;
let a, b, c;
let m, d;
let selection = "cubic";

//Sets the learning rate and the optimizer we will use
const learning_rate = 0.05;
const optimizer = tf.train.adam(learning_rate);

//The setup function sets up the canvas and initial code that must be run in the start.
function setup(){
    createCanvas(400, 400); //Makes a 400x400 canvas
    background(0); //Sets background to black

    //Adding the coefficients to the graph as variables based on 'selection'
    defineVars(selection);

}

function defineVars(selection){
    if (selection === "linear"){
        m = tf.variable(tf.scalar(random(1)));
        d = tf.variable(tf.scalar(random(1)));
    }
    else if(selection == "quadratic"){
        a = tf.variable(tf.scalar(random(1)));
        b = tf.variable(tf.scalar(random(1)));
        c = tf.variable(tf.scalar(random(1)));
    }
    else if(selection === "cubic"){
        e = tf.variable(tf.scalar(random(1)));
        f = tf.variable(tf.scalar(random(1)));
        g = tf.variable(tf.scalar(random(1)));
        h = tf.variable(tf.scalar(random(1)));
    };
}

function canvasToMap(x, y) {
    //A function I wrote myself to map x,y coordinates from the 400x400 canvas onto a grid
    //that goes from -1 to 1.

    if(x >= 200){
        mappedX = (x - 200)/200;
    }
    else if(x < 200){
        mappedX = x/200 - 1;
    }
    if(y >= 200){
        mappedY = - ((y-200)/200);
    }
    else if(y < 200){
        mappedY = 1 - y/200;
    }
    
    return [mappedX, mappedY];
}

function mapToCanvas(x, y) {
    //Another function I wrote to reverse the mapping from the (-1 to 1) back to the 400x400 canvas

    if(x >= 0) {
        canvasX = 200*x + 200;
    }
    else if(x < 0){
        canvasX = (x + 1) * 200;
    }
    if(y >= 0){
        canvasY = (1 - y) * 200;
    }
    else if (y < 0) {
        canvasY = ((-y)*200) + 200;
    }
    
    return [canvasX, canvasY];
}

function mousePressed(){
    //This function will take the mouse values (x and y) that come from user input
    //and add them to the arrays userX, userY.

    let [x, y] = canvasToMap(mouseX, mouseY);
    userX.push(x);
    userY.push(y);
}

function loss(labels, preds){
    //returns the mean square distance loss mean[(y - yhat)^2]

    return preds.sub(labels).square().mean();

}

function definePred(selection, x_tensor){
    if (selection === "linear"){
        y_hat = x_tensor.mul(m).add(d);
        return y_hat;
    }
    else if(selection == "quadratic"){
        y_hat = x_tensor.square().mul(a).add(x_tensor.mul(b)).add(c);
        return y_hat; 
    }
    else if(selection === "cubic"){
        y_hat = x_tensor.square().mul(x_tensor).mul(e).add(x_tensor.square().mul(f)).add(x_tensor.mul(g)).add(h);
        return y_hat;
    };
}

function predict(x){
    //Takes in a value x and returns the corresponding y value based on the current
    //coefficients of the equation

    //x needs to be a 1d tensor
    x_tensor = tf.tensor1d(x);
    y_hat = definePred(selection, x_tensor);
    

    return y_hat;
}


function draw(){
    //This is the p5.js function that runs on repeat and is responsible for 
    //consistently updating the canvas with latest results etc

    tf.tidy(() => {
        //Manages the extra tensors and tidies them up
        //Used for the training loop

        if (userX.length > 0){
            const y_tensor = tf.tensor1d(userY); //Converts userY to tensor format
            optimizer.minimize(() => loss(predict(userX), y_tensor)); //This is training step which
            //will run every time the draw loop runs and there is a user input
        }
    });

    background(0); //resets background to black
    strokeWeight(8); //How large the dots should be
    stroke(255); //Sets color of dots to white

    for (let i=0; i<userX.length; i++){
        let [px, py] = mapToCanvas(userX[i], [userY[i]]); //Gets each point
        point(px, py); //Plots each point
    };

    tf.tidy(() =>{
        const curveX = [];
        for (let i = -1; i<1; i+=0.05) {
            curveX.push(i); //Creates datapoints used to plot the curveX
        };

        curveY = predict(curveX); //Gets the model's latest predictions
        curveY_data = curveY.dataSync(); //Extracts the data

        beginShape(); //p5.js function that starts plotting a shape
        noFill(); //Do not want the shape to be filled with color
        stroke(255); //Want shape to be white in color
        strokeWeight(4); //Size of points along line
        for (let i = 0; i < curveX.length; i++){
            [x, y] = mapToCanvas(curveX[i], curveY_data[i]); //Extract a point
            vertex(x, y); //Plot the point to the canvas
        };
        endShape(); //p5.js function that ends plotting of a shape


    });
}
