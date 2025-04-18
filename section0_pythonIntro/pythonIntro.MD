
# TouchDesigner Python: Educational Guide

This guide covers core TouchDesigner Python concepts and how to create reusable, parameterized components using Python classes, `TDFunctions`, and custom properties.

---

## ✅ Using `StorageManager` vs. Implicit Storage

TouchDesigner objects have an implicit `.storage` dictionary that automatically remembers values across file saves. However, for explicit control and organization, `StorageManager` is preferred.

### Example:
```python
from storage import StorageManager

# Attach StorageManager to a component
self.default_params_stored = StorageManager(self, ownerComp, defaultParams)
```

> 🔁 TouchDesigner **remembers** changes to parameters across sessions, but `StorageManager` gives more structure for saved presets/configs.

---

## ✅ Creating Custom Parameters with `TDFunctions`

You can add custom properties with TDFunctions and make them reactive to Python attributes.

### Example:
```python
TDF.createProperty(
    self,
    name='dodivisions',
    value=1,
    dependable=True,
    readOnly=False
)
```

---

## ✅ Reactive Python Properties with TouchDesigner Params

To make your Python attributes reactive (e.g. `box.par.ry = self.rot_y`), use `@property`.

### Example:
```python
@property 
def rot_y(self):
    return self._rot_y

@rot_y.setter
def rot_y(self, value):
    print(f'Setting ry from {self._rot_y} to {value}')
    self._rot_y = value
```

---

## ✅ Working with Menus in Parameters

### Changing an existing menu parameter's values dynamically:

```python
g = op('../scatterPlot')
g.par.Colorgroup.menuNames = ['a', 'b', 'c']
```

You don't always need `TDF.parMenu()` unless you're assigning a `menuSource`. You can directly update `.menuNames` and `.menuLabels`.

---

## ✅ Creating New Menu Parameters

To create a menu-type parameter from scratch:

```python
page = comp.appendCustomPage('estetics')
page.appendStrMenu('ColorGroup', label='Color Group')
comp.par.ColorGroup.menuNames = ['red', 'green', 'blue']
```

You can skip `TDF.parMenu()` unless you're assigning the result to a menuSource object.

---

## ✅ Parameter Execute DAT: Hooking to Interactions

### Common callbacks:
```python
def onPulse(par):
    if par.name == "Applydebug":
        self.ApplyDebug()
```

- Use `.par.name` to detect the action.
- `SP = me.parent()` is often a reference to your main component.

---

## ✅ Component Class Example: Scatterplot Generator

```python
class ScatterplotGenerator:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        for item in defaultParams:
            TDF.createProperty(
                self,
                name=item['name'],
                value=item['value'],
                readOnly=item.get('readOnly', False),
                dependable=item.get('dependable', True)
            )

    def RestoreDefaultParameters(self):
        for param in defaultParams:
            self.ownerComp.par[param['name']] = param['value']
```

---

## ✅ Component Class Example: 3D Box Generator

```python
class Box(Geometry):
    def __init__(self, geoPath):
        super().__init__(geoPath)
        self.size_x = 0.2
        self._rot_y = 45
        self._dodivs = TDF.createProperty(self, 'dodivisions', value=1)

    def create(self):
        super().create()
        box = op(f"{self.geoPath}/geo").create(boxSOP, 'box')
        box.par.ry = self.rot_y
        box.par.sizex = self.size_x
        box.par.dodivs = self.dodivisions
```

---

## ✅ Naming Practices

- `SP`: fine for short, local variables.
- Better for clarity: `Scatterplot`, `scatterComp`, `scatterUI`, or `BlueprintScatter`.

---

## References

- [TouchDesigner Python Docs](https://docs.derivative.ca/Working_with_OPs_in_Python)
- [Custom Parameters](https://docs.derivative.ca/Custom_Parameters)
- [StorageManager Examples](https://docs.derivative.ca/StorageManager_Class)

---

🧠 Designed for modular, readable, reusable components in your TouchDesigner projects!


# Code implementation 
```python
from tdClasses import Box

# Create and set up box
box = Box(geoPath="/python_intro")
box.create()

# Store the box in the global context using mod.myGlobals
mod.myGlobals.box = box

# Access the operator and modify its parameters based on the box's properties
box_op = op('/python_intro/geo/box')
box_op.par.sizex = box.size_x  # Update the operator parameter sizex
box_op.par.ry = box.rot_y     # Update the operator parameter ry (if set in box)
box_op.par.dodivs = box.dodivisions  # Update the operator parameter dodivs (if set in box)

# Print to confirm changes
print(f"Operator sizex: {box_op.par.sizex}")
print(f"Operator ry: {box_op.par.ry}")
print(f"Operator dodivs: {box_op.par.dodivs}")

```