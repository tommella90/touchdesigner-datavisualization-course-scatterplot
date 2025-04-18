# TouchDesigner Extensions, Attributes, and Properties

## 📌 WHAT IS AN EXTENSION

An **Extension** in TouchDesigner is a **custom Python class** attached to a component. It allows you to encapsulate logic, methods, and data inside a reusable and clean structure.

```python
class ScatterplotGenerator:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.B = 1  # promoted attribute
        self.a = 0 # non-promoted attribute
```

- Attached via the `.ext` attribute (e.g., `op('scatterplot').ext`)
- Not a standard Python object — it’s a TouchDesigner-specific reference
- It allows your component to have custom logic, exposed through a familiar Python API

> 🔹 You can access the extension class via `op('yourComponent').ext`  
> 🔹 Promoting the extension makes its public interface accessible directly from `op('yourComponent')`

---

## 🚀 EXTENSIONS AND PROMOTING ATTRIBUTES AND METHODS

### 🔸 Attributes

| Term | Description | Example |
|------|-------------|---------|
| **Attribute** | A regular Python variable attached to an object | `self.baseColorList`, `self.ownerComp` |
| **Property**  | A variable created with `TDF.createProperty()` that integrates with TouchDesigner’s UI | See example below |

Attributes are like internal variables, whereas properties are "exposed" variables, useful in UI or scripting.

```python
TDF.createProperty(self, 'BaseColor',
    value=self.baseColorList[self.ColorIndex],
    readOnly=True,
    dependable=True)
```

Access examples:

| Object | What it represents | Example |
|--------|---------------------|---------|
| `self.BaseColor` | The actual value | `print(self.BaseColor)` returns `[1, 0, 0]` |
| `self._BaseColor` | Internal controller for the property | `self._BaseColor.val = [0, 1, 0]` |

Think of it as:

| What | Analogy |
|------|---------|
| `self.BaseColor` | Lightbulb |
| `self._BaseColor` | The dimmer or switch |

---

### 🔸 Promotion: UI & Accessibility

| Concept | Python “private” | TouchDesigner “not promoted” |
|--------|-------------------|-------------------------------|
| **Access control** | Just a naming hint (`_myVar`) | Actually hides from component UI |
| **Enforcement** | Not enforced | Hides from parameter window unless explicitly promoted |
| **Purpose** | Signal to devs not to use it | Clean UI, expose only what’s needed |

- Non-promoted: `self.myVar` → Access via `op('comp').ext.MyExtension.myVar`
- Promoted: `self.MyVar` → Access via `op('comp').MyVar`

> ✅ Promotion is about **exposing to UI**, not restricting access.

---

### 🔸 Promoting Methods

Yes, **methods** can also be promoted — not just attributes.

| What | Description |
|------|-------------|
| **Promoted Method** | A method made accessible directly via `op('comp').MyFunction()` |
| **Non-promoted Method** | Accessed through `op('comp').ext.MyExtension.MyFunction()` |

You typically promote a method via the Component’s "Promote Extension" parameter in the UI. Once done, it behaves like a built-in method.

---

## ⚙️ DIFFERENCE: ATTRIBUTE vs PROPERTY

| Term | What It Is | Access | Usage |
|------|------------|--------|-------|
| **Attribute** | Regular Python variable | Read/write freely | Internal logic |
| **Property** | Created using `TDF.createProperty()` | Shown in UI, optionally read-only/dependable | Exposed UI control |

Use `TDF.createProperty()` when:
- You want the value in the Parameter Window
- You want the value to be dependable (reactive to changes)
- You need read-only status or UI interaction

---

## 🧠 Summary

- Use **attributes** for internal variables.
- Use **properties** for exposed variables, especially when working with TouchDesigner’s UI.
- Use **extensions** to organize logic in a reusable, clean, and interactive way.
- Promote attributes/methods when you want them accessible directly from `op('yourComponent')`.

---

